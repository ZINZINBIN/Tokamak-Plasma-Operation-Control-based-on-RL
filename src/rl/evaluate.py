# optimization process for agent
import torch
import torch.nn as nn
import numpy as np
import gc
from typing import Optional, List, Literal
from src.rl.utility import InitGenerator
from itertools import count
from src.rl.env import NeuralEnv

def evaluate_ddpg(
    env : NeuralEnv,
    init_generator : InitGenerator,
    policy_network : nn.Module, 
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
    
    # reset ou noise and current state from env
    state = env.reset()
    state_list.append(init_state)
    action_list.append(init_action)

    for t in count():
        state = state.to(device)
        policy_network.eval()
        action = policy_network(state)

        env_input_action = action
        _, reward, done, _ = env.step(env_input_action)
        
        if state.ndim == 3:
            state = state.squeeze(0)
        
        if action.ndim == 3:
            action = action.squeeze(0)
        
        reward_list.append(reward.detach().cpu().numpy())
        state_list.append(state.detach().cpu().numpy())
        action_list.append(action.detach().cpu().numpy())
        
        if not done:
            next_state = env.get_state()
            state = next_state
        else:
            next_state = None
            break

    # memory cache delete
    gc.collect()

    # torch cache delete
    torch.cuda.empty_cache()

    print("\nTokamak Plasma Operation Evaluation Process done..!")
    env.close()
    
    return state_list, action_list, reward_list