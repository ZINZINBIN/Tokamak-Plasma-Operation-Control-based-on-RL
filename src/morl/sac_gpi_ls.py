''' 
    ==============================================================================================================================================
    SAC based multi-objective reinforcement learning algorithm

    The baseline algorithm is referred from the paper "A two-stage multi-objective deep reinforcement learning framework". 
    In this paper, two different algorithms are used for multi-objective task.
        (1) Multi-policy SAC algorithm
        (2) Multi-objective Convariance Matrix Adaptation Evolution Strategy
    Algorithm (1) is to aim learing the policy networks with linear scalarization method, shared low-level layers, and replay buffer sharing.
    These methods make the training process efficient and reduce the computational cost by sharing the weights and explorations obtained from 
    different policies. Collaborative learning is applied with respect to the each policy that is trained by different preference.
    
    Algorithm (2) is to aim optimizing the policy-independent parameters as a fine-tunning process. By doing this, it is possible to approach a 
    dense and uniform estimation of the Pareto frontier. 
    =============================================================================================================================================
    
    In this code, we implement Generalized Policy Improvement Linear Suport (GPI-LS) algorithm for SAC version.
    By computing corner weights and optimizing the policy based on corner weights, we can obtain a set of 
    
    Reference
    - Paper
        (1) A Two-Stage Multi-Objective Deep Reinforcement Learning Framework, Diqi Chen et al, ECAI, 2020
        (2) A pratical guide to multi-objective reinforcement learning and planning, Conor F.Hayes et al, 2022
    - Code
        (1) https://github.com/LucasAlegre/morl-baselines/tree/main
        (2) https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/multi_policy/gpi_pd/gpi_pd_continuous_action.py
        (3) https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/multi_policy/linear_support/linear_support.py
    - Extra
        (1) Multi-objective reinforcement learning : https://github.com/kevin880987/Multi-Objective-Reinforement-Learning
        (2) Multi-objective optimization and multi-task learning : https://github.com/xiaofangxd/Multi-objective-optimization-and-Multi-task-Learning
'''
import torch
import torch.nn as nn
import numpy as np
import gc, os, random
from tqdm.auto import tqdm
from typing import Optional, List, Literal, Dict, Union
from src.rl.sac import TwinnedQNetwork, GaussianPolicy, update_policy, update_policy_PER, train_sac
from src.rl.buffer import Transition, ReplayBuffer
from src.rl.PER import PER
from src.rl.utility import InitGenerator
from itertools import count
from src.rl.env import NeuralEnv
from src.rl.actions import smoothness_inducing_regularizer, add_noise_to_action
from src.morl.utility import random_weights, policy_evaluation

# for solving the vertex enumerate problem
import cdd

def compute_corner_weights(ccs : List, num_objectives : int):
    corner_weights = []
    A = np.vstack(ccs)
    A = np.round_(A, decimals = 4)
    A = np.concatenate((A, -np.ones(A.shape[0]).reshape(-1,1)), axis = 1)
    
    A_plus = np.ones(A.shape[1]).reshape(1,-1)
    A_plus[0,-1] = 0
    A = np.concatenate((A,A_plus), axis = 0)
    A_plus = -np.ones(A.shape[1]).reshape(1,-1)
    A_plus[0,-1] = 0
    A = np.concatenate((A,A_plus), axis = 0)
    
    for i in range(num_objectives):
        A_plus = np.zeros(A.shape[1]).reshape(1, -1)
        A_plus[0, i] = -1
        A = np.concatenate((A, A_plus), axis=0)
    
    b = np.zeros(len(ccs) + 2 + num_objectives)
    b[len(ccs)] = 1
    b[len(ccs) + 1] = -1
    
    def _compute_poly_vertices(A_:np.ndarray,b_:np.ndarray):
        b_ = b_.reshape((b_.shape[0], 1))
        mat = cdd.Matrix(np.hstack([b_, -A_]), number_type = "float")
        mat.rep_type = cdd.RepType.INEQUALITY
        P = cdd.Polyhedron(mat)
        g = P.get_generators()
        V = np.array(g)
        
        vertices = []
        
        for i in range(V.shape[0]):
            if V[i, 0] != 1:
                continue
            if i not in g.lin_set:
                vertices.append(V[i,1:])
        return vertices
    
    vertices = _compute_poly_vertices(A,b)
    for v in vertices:
        v = torch.from_numpy(v[:-1]).float()
        corner_weights.append(v)
    return corner_weights

def max_scalarized_value(w, ccs:List):
    if len(ccs) == 0:
        return None
    else:
        return np.max([np.dot(v,w) for v in ccs])

def gpi_ls_priority(w:np.ndarray, ccs : List, gpi_expanded_set : List):
    
    def best_vector(values, w):
        max_v = values[0]
        for i in range(1, len(values)):
            if values[i] @ w > max_v @ w:
                max_v = values[i]
        return max_v
    
    max_value_ccs = max_scalarized_value(w, ccs)
    max_value_gpi = best_vector(gpi_expanded_set, w)
    max_value_gpi = np.dot(max_value_gpi, w)
    priority = max_value_gpi - max_value_ccs
    return priority
    
def update_corner_weights(
    init_generator:InitGenerator,
    queue:List,
    env : NeuralEnv, 
    policy_network:GaussianPolicy, 
    ccs : List, 
    num_objectives : int,
    device : str
    ):
    
    if len(ccs) > 0:
        W_corner = compute_corner_weights(ccs, num_objectives)
        queue.clear()
        
        # policy evaluation
        gpi_expanded_set = [policy_evaluation(init_generator, env, policy_network, wc, 4, 32, device) for wc in W_corner]
        
        for wc in W_corner:
            priority = gpi_ls_priority(wc, ccs, gpi_expanded_set)
            queue.append((priority, wc))
        
        if len(queue) > 0:
            queue.sort(key = lambda t : t[0], reverse = True)
        
            if queue[0][0] == 0.0:
                random.shuffle(queue)
                
    if len(queue) == 0:
        return None
    else:
        next_w = queue.pop(0)[1]
        return next_w

def remove_obsolete_weight(new_value, ccs : List, queue : List):
    W_del = []
    
    if len(ccs) == 0:
        return W_del
    
    inds_remove = []
    
    for i, (priority, cw) in enumerate(queue):

        if torch.dot(cw, new_value) > max_scalarized_value(cw, ccs):
            W_del.append(cw)
            inds_remove.append(i)
    
    for i in reversed(inds_remove):
        queue.pop(i)
    
    return W_del

def remove_obsolete_values(value, visited_weights : List, weight_support:List, ccs : List, policy_list : List):
    removed_indx = []
    for i in reversed(range(len(ccs))):
        weights_optimal = [w for w in visited_weights if np.dot(ccs[i],w) == max_scalarized_value(w, ccs) and np.dot(value, w) < np.dot(ccs[i], w)]

        if len(weights_optimal) == 0:
            removed_indx.append(i)
            ccs.pop(i)
            weight_support.pop(i)
            policy_list.pop(i)
            
    return removed_indx

def is_dominated(value, visited_weights : List, ccs : List):
    
    if len(ccs) == 0:
        return False
    
    for w in visited_weights:
        if np.dot(value, w) >= max_scalarized_value(w, ccs):
            return False
        
    return True

def add_solution(value, weight, visited_weights : List, weight_support : List, ccs : List, policy_save_dir : str,  policy_list : List):
    
    visited_weights.append(weight)
    print(f"Adding value : {value} to CCS")
    
    if is_dominated(value, visited_weights, ccs):
        print(f"Value {value} is dominated. Removing the value")
        return [len(ccs)]
    
    removed_indx = remove_obsolete_values(value, visited_weights, weight_support, ccs, policy_list)
    
    ccs.append(value)
    weight_support.append(weight)
    policy_list.append(policy_save_dir)
    
    return removed_indx

def train_new_policy(
    env : NeuralEnv,
    init_generator : InitGenerator,
    memory : Union[ReplayBuffer, PER], 
    policy_network : GaussianPolicy, 
    q_network : TwinnedQNetwork, 
    target_q_network : TwinnedQNetwork,
    target_entropy : torch.Tensor,
    log_alpha : Optional[torch.Tensor],
    policy_optimizer : torch.optim.Optimizer,
    q1_optimizer : torch.optim.Optimizer,
    q2_optimizer : torch.optim.Optimizer,
    alpha_optimizer : Optional[torch.optim.Optimizer],
    criterion :nn.Module,
    batch_size : int = 128, 
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    min_value : float = -np.inf,
    max_value : float = np.inf,
    tau : float = 1e-2,
    num_episode : int = 256,  
    save_best : Optional[str] = None,
    save_last : Optional[str] = None,
    use_CAPS : bool = False,
    lamda_temporal_smoothness : float = 1.0,
    lamda_spatial_smoothness : float = 1.0,
    verbose : int = 8,
    ):
    
    T_MAX = 50

    if device is None:
        device = "cpu"

    best_reward = 0
    
    for i_episode in range(num_episode):
        
        # update new initial state and action
        init_state, init_action = init_generator.get_state()
        env.update_init_state(init_state, init_action)
        
        # reset ou noise and current state from env
        state = env.reset()
        
        reward_list = []
        state_list = []

        for t in range(T_MAX):
            
            # compute action
            policy_network.eval()
            action, _, _ = policy_network.sample(state.to(device))
            action = action.detach().cpu()
                        
            _, reward, done, _ = env.step(action)

            reward_list.append(reward.detach().cpu().numpy())
            reward = torch.tensor([reward])
            
            if not done:
                next_state = env.get_state()
            else:
                next_state = None

            # memory에 transition 저장
            memory.push(state, action, next_state, reward, done)

            # update state
            state = next_state
            
            if isinstance(memory, ReplayBuffer):
                q1_loss, q2_loss, policy_loss = update_policy(
                    memory, 
                    policy_network, 
                    q_network, 
                    target_q_network,
                    target_entropy,
                    log_alpha,
                    policy_optimizer,
                    q1_optimizer,
                    q2_optimizer,
                    alpha_optimizer,
                    criterion,
                    batch_size, 
                    gamma, 
                    device,
                    min_value,
                    max_value,
                    tau,
                    use_CAPS,
                    lamda_temporal_smoothness,
                    lamda_spatial_smoothness
                )
                
            elif isinstance(memory, PER):
                q1_loss, q2_loss, policy_loss = update_policy_PER(
                    memory, 
                    policy_network, 
                    q_network, 
                    target_q_network,
                    target_entropy,
                    log_alpha,
                    policy_optimizer,
                    q1_optimizer,
                    q2_optimizer,
                    alpha_optimizer,
                    criterion,
                    batch_size, 
                    gamma, 
                    device,
                    min_value,
                    max_value,
                    tau
                )
                
            
            # update state list
            if state is not None:
                state_list.append(state[:,-1,:].unsqueeze(0).numpy())

            if done or t > T_MAX:
                break
        
        max_reward = np.max(reward_list)
        min_reward = np.min(reward_list)
        mean_reward = np.mean(reward_list)
        
        if i_episode % verbose == 0:
            print(r"| episode:{:05d} | duration:{:04d} | reward - mean: {:.2f}, min: {:.2f}, max: {:.2f}".format(i_episode+1, t + 1, mean_reward, min_reward, max_reward))

        # memory cache delete
        gc.collect()

        # torch cache delete
        torch.cuda.empty_cache()
        
        # save weights
        torch.save(policy_network.state_dict(), save_last)
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(policy_network.state_dict(), save_best)

    memory.clear()

    return q_network, policy_network
    
def train_sac_linear_support(
    env : NeuralEnv,
    init_generator : InitGenerator,
    memory : Union[ReplayBuffer, PER], 
    policy_network : GaussianPolicy, 
    q_network : TwinnedQNetwork, 
    target_q_network : TwinnedQNetwork,
    target_entropy : torch.Tensor,
    log_alpha : Optional[torch.Tensor],
    policy_optimizer : torch.optim.Optimizer,
    q1_optimizer : torch.optim.Optimizer,
    q2_optimizer : torch.optim.Optimizer,
    alpha_optimizer : Optional[torch.optim.Optimizer],
    criterion :nn.Module,
    batch_size : int = 128, 
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    min_value : float = -np.inf,
    max_value : float = np.inf,
    tau : float = 1e-2,
    num_episode : int = 256,  
    verbose : int = 8,
    verbose_policy : int = 8,
    save_dir : Optional[str] = None,
    scaler_0D = None,
    use_CAPS : bool = False,
    lamda_temporal_smoothness : float = 1.0,
    lamda_spatial_smoothness : float = 1.0,
    max_gpi_ls_iters : int = 32,
    num_objectives : int = 2,
    tag : str = "",
    seed : int = 42,
    ):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    policy_set = []
    visited_weights = []
    weight_support = []
    ccs = []
    queue = []
    
    def _init_weights(dim: int, is_random : bool = True, seed : int = 42):
        if is_random:
            return list(random_weights(dim, n = 8, seed = seed))
        else:
            return list(torch.eye(dim, dtype=torch.float32))
    
    for w in _init_weights(num_objectives, True, seed):
        queue.append((float("inf"), w))
    
    for gpi_ls_iter in tqdm(range(max_gpi_ls_iters)):
        
        # directory info
        save_best = os.path.join(save_dir, "{}_{}_best.pt".format(tag, gpi_ls_iter))
        save_last = os.path.join(save_dir, "{}_{}_last.pt".format(tag, gpi_ls_iter))
        
        # update cornder weight first
        w_next = update_corner_weights(init_generator, queue, env, policy_network, ccs, num_objectives, device)
        
        # initialize the networks
        policy_network.initialize()
        q_network.initialize()
        target_q_network.initialize()
        
        # update reward sender with initial weight
        env.reward_sender.update_target_weight(w_next)
         
        # find the optimal policy and value newtork with given initial weight
        print("="*20, " New policy with updated weight ({:03d}/{:03d})".format(gpi_ls_iter+1, max_gpi_ls_iters), "="*20)
        print("Updated weight vector: ", w_next)
        
        _, _ = train_new_policy(
            env,
            init_generator,
            memory,
            policy_network,
            q_network,
            target_q_network,
            target_entropy,
            log_alpha,
            policy_optimizer,
            q1_optimizer,
            q2_optimizer,
            alpha_optimizer,
            criterion,
            batch_size,
            gamma,
            device,
            min_value,
            max_value,
            tau,
            num_episode,
            save_best,
            save_last,
            use_CAPS,
            lamda_temporal_smoothness,
            lamda_spatial_smoothness,
            verbose_policy
        )
        
        value = policy_evaluation(init_generator, env, policy_network, w_next, 4, 32, device)
        
        # add new policy and value vector with removing the dominated value vector set
        add_solution(value, w_next,visited_weights, weight_support, ccs, save_best, policy_set)
        
        # condition
        if len(queue) == 0:
            break
        
        if gpi_ls_iter % verbose == 0:
            
            print("="*80)
            print("GPI-LS : weight update process ({}/{})".format(gpi_ls_iter + 1, max_gpi_ls_iters))
            for i, (target_value, idx) in enumerate(zip(env.reward_sender.targets_value, env.reward_sender.target_cols_indices)):
                key = env.reward_sender.targets_cols[i]
                wi = w_next[i]
                print("Target features:{:10} | Target value:{:.2f} | Reward:{:.2f} | Weight:{:.2f}".format(key,target_value,value[i],wi))
                print("# of queue: {}".format(len(queue)))
            print("="*80)
               
    print("Generalized Policy Improvement Linear Support : SAC training process done..!")
    return weight_support, ccs, policy_set