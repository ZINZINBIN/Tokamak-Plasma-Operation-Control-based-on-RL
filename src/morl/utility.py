from typing import List, Optional
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.rl.sac import GaussianPolicy
from src.rl.env import NeuralEnv
from src.rl.utility import InitGenerator
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
from pymoo.util.ref_dirs.energy_layer import LayerwiseRieszEnergyReferenceDirectionFactory
from pymoo.util.ref_dirs.reduction import ReductionBasedReferenceDirectionFactory
from pymoo.util.reference_direction import MultiLayerReferenceDirectionFactory
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

def hypervolume(ref_point: np.ndarray, points: List[npt.ArrayLike]):
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point (from Pymoo).

    Args:
        ref_point (np.ndarray): Reference point (number of objectives)
        points (List[np.ndarray]): List of value vectors (CCS)

    Returns:
        float: Hypervolume metric
    """
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)

def get_reference_directions(name, *args, **kwargs):
    REF = {
        "uniform": UniformReferenceDirectionFactory,
        "das-dennis": UniformReferenceDirectionFactory,
        "energy": RieszEnergyReferenceDirectionFactory,
        "multi-layer": MultiLayerReferenceDirectionFactory,
        "layer-energy": LayerwiseRieszEnergyReferenceDirectionFactory,
        "reduction": ReductionBasedReferenceDirectionFactory,
    }

    if name not in REF:
        raise Exception("Reference directions factory not found.")

    return REF[name](*args, **kwargs)()

def random_weights(dim: int, n: int = 1, dist: str = "dirichlet", seed: Optional[int] = None, rng: Optional[np.random.Generator] = None):
    """Generate random normalized weight vectors from a Gaussian or Dirichlet distribution alpha=1.

    Args:
        dim: size of the weight vector
        n : number of weight vectors to generate
        dist: distribution to use, either 'gaussian' or 'dirichlet'. Default is 'dirichlet' as it is equivalent to sampling uniformly from the weight simplex.
        seed: random seed
        rng: random number generator
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if dist == "gaussian":
        w = rng.standard_normal((n, dim))
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1, keepdims=True)
    elif dist == "dirichlet":
        w = rng.dirichlet(np.ones(dim), n)
    else:
        raise ValueError(f"Unknown distribution {dist}")

    if n == 1:
        return w[0]
    return w

def policy_evaluation(
    init_generator:InitGenerator,
    env : NeuralEnv, 
    policy_network:GaussianPolicy, 
    weight : List,
    num_episode : int,
    T_MAX : int,
    device : str,
    ):
    
    mean_vec_return = []
    env.reward_sender.update_target_weight(weight)
    policy_network.eval()
    policy_network.to(device)
    
    for i_episode in range(num_episode):
        
        # update new initial state and action
        init_state, init_action = init_generator.get_state()
        env.update_init_state(init_state, init_action)
        
        # reset ou noise and current state from env
        state = env.reset()
        
        reward_list = []
        gamma = 1.0

        for t in range(T_MAX):
            
            # compute action
            policy_network.eval()
            action, _, _ = policy_network.sample(state.to(device))
            action = action.detach().cpu()
                        
            state, _, done, _ = env.step(action)
            vec_reward = env.reward_sender.compute_vectorized_reward(state)

            reward_list.append(vec_reward.detach() * gamma)
            
            gamma *= env.gamma
            
            if not done:
                next_state = env.get_state()
            else:
                next_state = None

            # update state
            state = next_state
            
            if done or t > T_MAX:
                break
        
        reward = torch.stack(reward_list).mean(axis = 0).view(-1,)
        mean_vec_return.append(reward)
    
    mean_vec_return = torch.mean(torch.stack(mean_vec_return), dim = 0)
    
    return mean_vec_return

def plot_pareto_front(
    init_generator : InitGenerator,
    env : NeuralEnv,
    policy_network : GaussianPolicy, 
    policy_set : List, 
    weight_support : List, 
    target_list_plot : List,
    target_list_total : List,
    device : str,
    save_file:str,
    reference_set : Optional[str] = None,
    num_episode : int = 8,
    T_MAX : int = 32,
    ):
    
    fig, ax = plt.subplots(1,1,figsize = (8,8))
    
    if reference_set is not None:
        policy_network.initialize()
        policy_network.to(device)
        policy_network.load_state_dict(torch.load(reference_set))
        reference_vec_reward = policy_evaluation(init_generator, env, policy_network, [0.5, 0.5], num_episode, T_MAX, device).detach().cpu().view(-1,).numpy()
    else:
        reference_vec_reward = None
        
    reward_set = []
    
    for path, weight in zip(policy_set, weight_support):
    
        # initialize
        policy_network.initialize()
        
        # load parato-Optimal policy
        policy_network.to(device)
        policy_network.load_state_dict(torch.load(path))
            
        # compute mean reward as a vector
        vec_reward = policy_evaluation(init_generator, env, policy_network, weight, num_episode, T_MAX, device)
        
        reward_set.append(vec_reward.detach().cpu().view(1,-1).numpy())
        
    reward_set = np.vstack(reward_set).reshape(-1, len(target_list_total))
    label_target = []
    indices = []
    
    for idx in range(len(target_list_total)):
        
        if target_list_total[idx] not in target_list_plot:
            continue
        
        label = target_list_total[idx]
        label_target.append(label)
        indices.append(idx)    
    
    reward_set = reward_set[:, indices].reshape(-1, len(indices))
    
    # figure 1 : plot the pareto-frontier set with respect to the reward for each objective
    ax.scatter(reward_set[:,0], reward_set[:,1], c='b', label = "GPI-LS")
    
    if reference_vec_reward is not None:
        ax.scatter(reference_vec_reward[indices][0], reference_vec_reward[indices][1],c='r', label = "Reference")
    
    ax.set_xlabel("Expected return (objective:{})".format(target_list_plot[0]))
    ax.set_ylabel("Expected return (objective:{})".format(target_list_plot[1]))
    ax.legend(loc = 'upper right')
    ax.set_title('Pareto frontier with GPI-LS and Reference (Target:{} and {})'.format(target_list_plot[0], target_list_plot[1]))
    
    fig.tight_layout()
    plt.savefig(save_file)