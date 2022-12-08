# optimization process for agent
import torch
import torch.nn as nn
import numpy as np
import gc
from typing import Optional, List, Literal
from src.rl.buffer import Transition, ReplayBuffer

# update policy
def update_policy(
    memory : ReplayBuffer, 
    policy_network : nn.Module, 
    value_network : nn.Module, 
    target_policy_network : nn.Module,
    target_value_network : nn.Module,
    value_optimizer : torch.optim.Optimizer,
    policy_optimizer : torch.optim.Optimizer,
    criterion :Optional[nn.Module] = None,
    batch_size : int = 128, 
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    min_value : float = -np.inf,
    max_value : float = np.inf,
    tau : float = 1e-2
    ):

    policy_network.train()
    value_network.train()

    if len(memory) < batch_size:
        return None, None

    if device is None:
        device = "cpu"
    
    if criterion is None:
        criterion = nn.SmoothL1Loss(reduction='mean') # Huber Loss
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # 최종 상태가 아닌 경우의 mask
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

    state_batch_ = state_batch.clone()

    # update value network
    # Q = r + r'* Q'(s_{t+1}, J(a|s_{t+1}))
    # Loss[y - Q] -> update value network
    next_q_values = torch.zeros((batch_size,1), device = device)
    next_q_values[non_final_mask] = target_value_network(non_final_next_states, target_policy_network(non_final_next_states).detach()).detach()
    
    bellman_q_values = reward_batch.unsqueeze(1) + gamma * next_q_values
    bellman_q_values = torch.clamp(bellman_q_values, min_value, max_value).detach()
    
    q_values = value_network(state_batch, action_batch)
    value_loss = criterion(q_values, bellman_q_values)

    value_optimizer.zero_grad()
    value_loss.backward()    
    value_optimizer.step()

    # update policy network 
    # sum of Q-value -> grad Q(s,a) * grad J(a|s) -> update policy
    policy_loss = value_network(state_batch, policy_network(state_batch))
    policy_loss = -policy_loss.mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # gradient clipping for value_network and policy_network
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1,1) 
    
    for param in value_network.parameters():
        param.grad.data.clamp_(-1,1) 

    # target network soft tau update
    for target_param, param in zip(target_value_network.parameters(), value_network.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

    for target_param, param in zip(target_policy_network.parameters(), policy_network.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

    return value_loss.item(), policy_loss.item()

def train_ddpg(
    env,
    memory : ReplayBuffer, 
    policy_network : nn.Module, 
    value_network : nn.Module, 
    target_policy_network : nn.Module,
    target_value_network : nn.Module,
    policy_optimizer : torch.optim.Optimizer,
    value_optimizer : torch.optim.Optimizer,
    value_loss_fn :Optional[nn.Module] = None,
    batch_size : int = 128, 
    gamma : float = 0.99, 
    device : Optional[str] = "cpu",
    min_value : float = -np.inf,
    max_value : float = np.inf,
    tau : float = 1e-2,
    num_episode : int = 256,  
    verbose : int = 8
    ):

    value_network.train()
    policy_network.train()

    target_value_network.eval()
    target_policy_network.eval()

    if device is None:
        device = "cpu"

    episode_durations = []
    reward_list = []
    ou_noise = OUNoise(env.action_space)

    for i_episode in tqdm(range(num_episode)):
        env.reset()
        ou_noise.reset()
        state = get_screen(env)
        mean_reward = []

        for t in count():
            state = state.to(device)
            policy_network.eval()
            action = policy_network(state).detach()
            env_input_action = ou_noise.get_action(action.detach().cpu().numpy()[0,0], t)
            _, reward, done, _ = env.step(env_input_action)

            mean_reward.append(reward)
            reward = torch.tensor([reward], device = device)

            if not done:
                next_state = get_screen(env)

            else:
                next_state = None
            
            # memory에 transition 저장
            memory.push(state, action, next_state, reward, done)

            state = next_state

            value_loss, policy_loss = update_policy(
                memory,
                policy_network,
                value_network,
                target_policy_network,
                target_value_network,
                value_optimizer,
                policy_optimizer,
                value_loss_fn,
                batch_size,
                gamma,
                device,
                min_value,
                max_value,
                tau
            )

            if done:
                episode_durations.append(t+1)
                mean_reward = np.mean(mean_reward)
                break

        if i_episode % verbose == 0:
            print("episode : {}, duration : {} and mean reward : {:.2f}".format(i_episode+1, t + 1, mean_reward))

        reward_list.append(mean_reward) 

        # memory cache delete
        gc.collect()

        # torch cache delete
        torch.cuda.empty_cache()

    print("training policy network and target network done....!")
    env.close()

    return episode_durations, reward_list