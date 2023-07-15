# SAC algorithm for continuous action space
# The paper : Soft Actor-Critic Algorithms and Applications, https://arxiv.org/pdf/1812.05905.pdf
# reference : https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954
# code : https://github.com/toshikwa/soft-actor-critic.pytorch/blob/master/code/agent.py
# Actor-critic based model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gc, time
from tqdm.auto import tqdm
from typing import Optional, List, Literal, Dict, Union
from src.rl.buffer import Transition, ReplayBuffer
from src.rl.PER import PER
from src.rl.utility import InitGenerator, initialize_weights
from itertools import count
from src.rl.env import NeuralEnv
from src.rl.actions import smoothness_inducing_regularizer, add_noise_to_action

# Encoder : plasma state -> hidden vector
class Encoder(nn.Module):
    def __init__(self, input_dim : int, seq_len : int, pred_len : int, conv_dim : int = 32, conv_kernel : int = 3, conv_stride : int = 2, conv_padding : int = 1):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        
        # temporal convolution
        self.layer_1 = nn.Sequential(
            nn.Conv1d(in_channels = input_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels = conv_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
        )    
        
        self.feature_dim = conv_dim
                
        # output as sequential length = pred_len
        self.layer_2 = nn.Linear(
            self.compute_conv1d_output_dim(self.compute_conv1d_output_dim(seq_len, conv_kernel, conv_stride, conv_padding, 1), conv_kernel, conv_stride, conv_padding, 1),
            pred_len
        )
        
    def forward(self, x : torch.Tensor):
        
        # normalization
        x = F.normalize(x, dim = 0)
        
        # x : (B, T, D)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        
        # x : (B, D, T)
        if x.size()[2] != self.seq_len:
            x = x.permute(0,2,1)
        
        # x : (B, conv_dim, T')
        x = self.layer_1(x)
        
        # x : (B, conv_dim, pred_len)
        x = self.layer_2(x)
        
        # x : (B, pred_len, conv_dim)
        x = x.permute(0,2,1)
        
        return x
    
    def compute_conv1d_output_dim(self, input_dim : int, kernel_size : int = 3, stride : int = 1, padding : int = 1, dilation : int = 1):
        return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

# Actor : Gaussian policy network
class GaussianPolicy(nn.Module):
    def __init__(self, input_dim : int, seq_len : int, pred_len : int, mlp_dim : int, n_actions : int, log_std_min : float = -10, log_std_max : float = 1.0):
        super(GaussianPolicy, self).__init__()
        self.encoder = Encoder(input_dim, seq_len, pred_len)
        self.n_actions = n_actions
        
        self.mlp_dim = mlp_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        linear_input_dim = self.encoder.feature_dim

        self.mlp_mean = nn.Sequential(
            nn.Linear(linear_input_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, n_actions),
        )
        
        self.mlp_std = nn.Sequential(
            nn.Linear(linear_input_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, n_actions),
        )
        
    def forward(self, x : torch.Tensor):
        # x : (B, T, D)
        x = self.encoder(x) # x : (B, pred_len, linear_input_dim)
        
        # mu : (B, pred_len, n_action)
        mu = self.mlp_mean(x)
        
        # std : (B, pred_len, n_action)
        log_std = self.mlp_std(x)
        log_std = torch.clamp(log_std, min = self.log_std_min, max = self.log_std_max)
        
        return mu, log_std
    
    def sample(self, x : torch.Tensor): 
        mu, log_std = self.forward(x)
        std = log_std.exp()
        
        normal_dist = Normal(mu, std)
        xs = normal_dist.rsample()
        action = torch.tanh(xs)
        
        log_probs = normal_dist.log_prob(xs) - torch.log(1 - action.pow(2) + 1e-3)
        # ignore sequence axis
        log_probs = log_probs.sum(dim=1, keepdim = False)
        # compute entropy with conserving action dimension
        entropy = -log_probs.sum(dim=1, keepdim = True)
        
        return action, entropy, torch.tanh(mu)
    
    def initialize(self):
        self.apply(initialize_weights)
            
# Critic
class QNetwork(nn.Module):
    def __init__(self, input_dim : int, seq_len : int, pred_len : int, mlp_dim : int, n_actions : int):
        super(QNetwork, self).__init__()
        self.encoder = Encoder(input_dim, seq_len, pred_len)
        self.n_actions = n_actions
        self.mlp_dim = mlp_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + n_actions, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )
        
    def forward(self, state:torch.Tensor, action : torch.Tensor):
        state = self.encoder(state)
        x = torch.cat([state, action], dim = 2).mean(axis = 1)
        x = self.mlp(x)
        return x
    
class TwinnedQNetwork(nn.Module):
    def __init__(self, input_dim : int, seq_len : int, pred_len : int, mlp_dim : int, n_actions : int):
        super().__init__()
        self.Q1 = QNetwork(input_dim, seq_len, pred_len, mlp_dim, n_actions)
        self.Q2 = QNetwork(input_dim, seq_len, pred_len, mlp_dim, n_actions)
        
    def forward(self, state : torch.Tensor, action : torch.Tensor):
        
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        
        return q1, q2
    
    def initialize(self):
        self.apply(initialize_weights)

# update process for policy and q-network in SAC
def update_policy(
    memory : ReplayBuffer, 
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
    use_CAPS : bool = False,
    lamda_temporal_smoothness : float = 1.0, 
    lamda_spatial_smoothness : float = 1.0,
    ):
    
    if use_CAPS:
        loss_fn_CAPS = nn.SmoothL1Loss(reduction='mean')
    else:
        loss_fn_CAPS = None
    
    # policy and q network => parameter update : True
    policy_network.train()
    q_network.train()
    
    # target network => parameter update : False
    target_q_network.eval()
    
    if len(memory) < batch_size:
        return None, None, None

    if device is None:
        device = "cpu"
    
    # replaybuffer
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    # masking the next state tensor for non-final state
    non_final_mask = torch.tensor(
        tuple(
            map(lambda s : s is not None, batch.next_state)
        ),
        device = device,
        dtype = torch.bool
    )
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    
    alpha = log_alpha.exp()

    # step 1. update Q-network parameters
    # Q = r + r'* Q'(s_{t+1}, J(a|s_{t+1}))
    # J = Loss[(Q - (r + r'*Q(s_{t+1}, a_{t+1})))^2]
    q1, q2 = q_network(state_batch, action_batch)
    
    with torch.no_grad():
        next_action_batch, next_entropy, _ = policy_network.sample(non_final_next_states)
        next_q1, next_q2 = target_q_network(non_final_next_states, next_action_batch)
        
        next_q = torch.zeros((batch_size,1), device = device)
        next_q[non_final_mask] = torch.min(next_q1, next_q2) + alpha.to(device) * next_entropy
    
    bellman_q_values = reward_batch.unsqueeze(1) + gamma * next_q
    bellman_q_values = torch.clamp(bellman_q_values, min_value, max_value)
    
    q1_loss = criterion(q1, bellman_q_values)
    q1_optimizer.zero_grad()
    q1_loss.backward()
    
    for param in q_network.Q1.parameters():
        param.grad.data.clamp_(-1,1) 
            
    q1_optimizer.step()
    
    q2_loss = criterion(q2, bellman_q_values)
    q2_optimizer.zero_grad()
    q2_loss.backward()
    
    for param in q_network.Q2.parameters():
        param.grad.data.clamp_(-1,1) 
        
    q2_optimizer.step()
    
    # step 2. update policy weights
    action_batch_sampled, entropy, _ = policy_network.sample(state_batch)
    q1, q2 = q_network(state_batch, action_batch)
    q = torch.min(q1, action_batch_sampled)
    
    policy_loss = torch.mean(q + entropy * alpha.to(device)) * (-1)
    
    if use_CAPS:
        mu,_,_ = policy_network.sample(state_batch[non_final_mask])
        next_mu,_,_ = policy_network.sample(non_final_next_states)
        near_mu = add_noise_to_action(mu)
        loss_CAPS = smoothness_inducing_regularizer(loss_fn_CAPS, mu, next_mu, near_mu, lamda_temporal_smoothness, lamda_spatial_smoothness)
        policy_loss += loss_CAPS
    else:
        pass
    
    policy_optimizer.zero_grad()
    policy_loss.backward()
    
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1,1) 
        
    policy_optimizer.step()
    
    # step 3. adjust temperature
    entropy_loss = (-1) * torch.mean(log_alpha.to(device) * (target_entropy.to(device) - entropy).detach())
    alpha_optimizer.zero_grad()
    entropy_loss.backward()
    alpha_optimizer.step()
    
    # step 4. update target network parameter
    for target_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    return q1_loss.item(), q2_loss.item(), policy_loss.item()

# update policy using PER 
def update_policy_PER(
    memory : PER, 
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
    tau : float = 1e-2
    ):
    
    # policy and q network => parameter update : True
    policy_network.train()
    q_network.train()
    
    # target network => parameter update : False
    target_q_network.eval()
    
    if memory.tree.n_entries < batch_size:
        return None, None, None

    if device is None:
        device = "cpu"
    
    # sampling from prioritized replay buffer
    transitions, indice, is_weight = memory.sample(batch_size)
    is_weight = torch.FloatTensor(is_weight).to(device)
    
    batch = Transition(*zip(*transitions))
    
    # masking the next state tensor for non-final state
    non_final_mask = torch.tensor(
        tuple(
            map(lambda s : s is not None, batch.next_state)
        ),
        device = device,
        dtype = torch.bool
    )
    
    # exception : if all next state are None which means that all states are final, then ignore this samples
    if non_final_mask.sum() == batch_size:
        return None, None, None
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    
    alpha = log_alpha.exp()

    # step 1. update Q-network parameters
    # Q = r + r'* Q'(s_{t+1}, J(a|s_{t+1}))
    # J = Loss[(Q - (r + r'*Q(s_{t+1}, a_{t+1})))^2]
    q1, q2 = q_network(state_batch, action_batch)
    
    with torch.no_grad():
        next_action_batch, next_entropy, _ = policy_network.sample(non_final_next_states)
        next_q1, next_q2 = target_q_network(non_final_next_states, next_action_batch)
        
        next_q = torch.zeros((batch_size,1), device = device)
        next_q[non_final_mask] = torch.min(next_q1, next_q2) + alpha.to(device) * next_entropy
    
    bellman_q_values = reward_batch.unsqueeze(1) + gamma * next_q
    bellman_q_values = torch.clamp(bellman_q_values, min_value, max_value)
    
    # step 1-1. priorities computation for PER
    prios = bellman_q_values.clone().detach() - torch.min(q1.clone().detach(), q2.clone().detach())
    prios = np.abs(prios.cpu().numpy())
    
    # update PER
    for batch_idx, idx in enumerate(indice):
        memory.update(idx, prios[batch_idx])
    
    # Q-network optimization
    q1_loss = criterion(q1, bellman_q_values) * is_weight
    q1_loss = q1_loss.mean()
    q1_optimizer.zero_grad()
    q1_loss.backward()
    
    for param in q_network.Q1.parameters():
        param.grad.data.clamp_(-1,1) 
            
    q1_optimizer.step()
    
    q2_loss = criterion(q2, bellman_q_values) * is_weight
    q2_loss = q2_loss.mean()
    q2_optimizer.zero_grad()
    q2_loss.backward()
    
    for param in q_network.Q2.parameters():
        param.grad.data.clamp_(-1,1) 
        
    q2_optimizer.step()
    
    # step 2. update policy weights
    action_batch_sampled, entropy, _ = policy_network.sample(state_batch)
    q1, q2 = q_network(state_batch, action_batch)
    q = torch.min(q1, action_batch_sampled)
    
    policy_loss = torch.mean(q + entropy * alpha.to(device)) * (-1)
    policy_optimizer.zero_grad()
    policy_loss.backward()
    
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1,1) 
        
    policy_optimizer.step()
    
    # step 3. adjust temperature
    entropy_loss = (-1) * torch.mean(log_alpha.to(device) * (target_entropy.to(device) - entropy).detach())
    alpha_optimizer.zero_grad()
    entropy_loss.backward()
    alpha_optimizer.step()
    
    # step 4. update target network parameter
    for target_param, param in zip(target_q_network.parameters(), q_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    return q1_loss.item(), q2_loss.item(), policy_loss.item()

def train_sac(
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
    save_best : Optional[str] = None,
    save_last : Optional[str] = None,
    scaler_0D = None,
    use_CAPS : bool = False,
    lamda_temporal_smoothness : float = 1.0,
    lamda_spatial_smoothness : float = 1.0
    ):
    
    T_MAX = 1024

    if device is None:
        device = "cpu"

    episode_durations = []
    min_reward_list = []
    max_reward_list = []
    mean_reward_list = []
    
    target_value_result = {}
    
    for key in env.reward_sender.targets_cols:
        target_value_result[key] = {
            "min" : [],
            "max" : [],
            "mean": [],
        }
    
    best_reward = 0
    best_episode = 0
    
    for i_episode in tqdm(range(num_episode), desc = "SAC algorithm training process"):
        
        start_time = time.time()
        
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
                episode_durations.append(t+1)
                
                max_reward = np.max(reward_list)
                min_reward = np.min(reward_list)
                mean_reward = np.mean(reward_list)
                
                state_list = np.squeeze(np.concatenate(state_list, axis = 1), axis = 0)
                
                if scaler_0D:
                    state_list = scaler_0D.inverse_transform(state_list)
                
                for idx, key in zip(env.reward_sender.target_cols_indices, env.reward_sender.targets_cols):
                    target_value_result[key]['min'].append(np.min(state_list[:,idx]))
                    target_value_result[key]['max'].append(np.max(state_list[:,idx]))
                    target_value_result[key]['mean'].append(np.mean(state_list[:,idx]))
                
                break
            
        end_time = time.time()
        
        if i_episode % verbose == 0:
            print(r"| episode:{} | duration:{} | reward - mean: {:.2f}, min: {:.2f}, max: {:.2f} | run time : {:.2f} | done : {}".format(i_episode+1, t + 1, mean_reward, min_reward, max_reward, end_time - start_time, done))

        min_reward_list.append(min_reward)
        max_reward_list.append(max_reward)
        mean_reward_list.append(mean_reward)

        # memory cache delete
        gc.collect()

        # torch cache delete
        torch.cuda.empty_cache()
        
        # save weights
        torch.save(policy_network.state_dict(), save_last)
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_episode = i_episode
            torch.save(policy_network.state_dict(), save_best)

    print("RL training process clear....!")
    env.close()
    
    episode_reward = {
        "min" : min_reward_list,
        "max" : max_reward_list,
        "mean" : mean_reward_list
    }

    return target_value_result, episode_reward


def evaluate_sac(
    env : NeuralEnv,
    init_generator : InitGenerator,
    policy_network : GaussianPolicy, 
    device : Optional[str] = "cpu",
    shot_num : Optional[int] = None
    ):
    
    policy_network.eval()

    if device is None:
        device = "cpu"

    episode_durations = []
    reward_list = []
    state_list = []
    action_list = []

    if shot_num:
        init_generator.shot_num = shot_num
        
    # update new initial state and action
    init_state, init_action = init_generator.get_state()
    env.update_init_state(init_state, init_action)
    
    # reset current state from env
    state = env.reset()
    state_list.append(env.get_state().squeeze(0))
    action_list.append(env.get_action().squeeze(0))

    for t in count():
        policy_network.eval()
        action, _, _ = policy_network.sample(state.to(device))
        action = action.detach().cpu()             
        _, reward, done, _ = env.step(action)

        if not done:
            next_state = env.get_state()
            state = next_state
        else:
            next_state = None
            break
        
        if state.ndim == 3:
            state = state.squeeze(0)
        
        if action.ndim == 3:
            action = action.squeeze(0)
    
        reward_list.append(reward.detach().cpu().numpy())
        state_list.append(state.detach().cpu().numpy())
        action_list.append(action.detach().cpu().numpy())

    # memory cache delete
    gc.collect()

    # torch cache delete
    torch.cuda.empty_cache()

    print("\nTokamak Plasma Operation Evaluation Process done..!")
    env.close()
    
    return state_list, action_list, reward_list
    