from src.rl.env import NeuralEnv
from src.nn_env.transformer import Transformer
from src.rl.rewards import RewardSender
from src.rl.utility import InitGenerator, preparing_initial_dataset
from src.rl.ddpg import Actor, Critic, train_ddpg, OUNoise
from src.rl.buffer import ReplayBuffer
from src.config import Config
import torch
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parsing():
    parser = argparse.ArgumentParser(description="training ddpg algorithms for tokamak plasma control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "DDPG")
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # scenario for training
    parser.add_argument("--shot_random", type = bool, default = True)
    parser.add_argument("--t_init", type = float, default = 0.0)
    parser.add_argument("--t_terminal", type = float, default = 10.0)
    parser.add_argument("--dt", type = float, default = 0.01)
    
    # DDPG training setup
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--num_episode", type = int, default = 1024)  
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--gamma", type = float, default = 0.995)
    parser.add_argument("--min_value", type = float, default = -10.0)
    parser.add_argument("--max_value", type = float, default = 10.0)
    parser.add_argument("--tau", type = float, default = 0.01)
    parser.add_argument("--verbose", type = int, default = 4)
    
    # predictor config
    parser.add_argument("--predictor_weight", type = str, default = "./weights/Transformer_seq16_dis4_best.pt")
    parser.add_argument("--seq_len", type = int, default = 16)
    parser.add_argument("--pred_len", type = int, default = 4)
    
    args = vars(parser.parse_args())

    return args

# torch device state
print("############### device setup ###################")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()
    
if __name__ == "__main__":
    
    # parsing
    args = parsing()
    
    tag = args['tag']
    save_dir = args['save_dir']
    batch_size = args['batch_size']
    num_episode = args['num_episode']
    seq_len = args['seq_len']
    pred_len = args['pred_len']
    t_init = args['t_init']
    lr = args['lr']
    gamma = args['gamma']
    min_value = args['min_value']
    max_value = args['max_value']
    tau = args['tau']
    verbose = args['verbose']
        
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
        
    config = Config()
    
    # 0D parameter
    cols_0D = config.DEFAULT_COLS_0D
    
    # else diagnostics
    cols_diag = config.DEFAULT_COLS_DIAG
    
    # control value / parameter
    cols_control = config.DEFAULT_COLS_CTRL
    
    # predictor
    model = Transformer(
        n_layers = config.TRANSFORMER_CONF['n_layers'], 
        n_heads = config.TRANSFORMER_CONF['n_heads'], 
        dim_feedforward = config.TRANSFORMER_CONF['dim_feedforward'], 
        dropout = config.TRANSFORMER_CONF['dropout'],        
        RIN = config.TRANSFORMER_CONF['RIN'],
        input_0D_dim = len(cols_0D),
        input_0D_seq_len = seq_len,
        input_ctrl_dim = len(cols_control),
        input_ctrl_seq_len = seq_len + pred_len,
        output_0D_pred_len = pred_len,
        output_0D_dim = len(cols_0D),
        feature_0D_dim = config.TRANSFORMER_CONF['feature_0D_dim'],
        feature_ctrl_dim = config.TRANSFORMER_CONF['feature_ctrl_dim'],
    )

    model.to(device)
    model.load_state_dict(torch.load(args['predictor_weight']))

    # reward 
    targets_dict = config.DEFAULT_TARGETS

    # reward
    reward_sender = RewardSender(targets_dict, total_cols = cols_0D)
    
    # environment
    env = NeuralEnv(predictor=model, device = device, reward_sender = reward_sender, seq_len = seq_len, pred_len = pred_len, t_terminal = args['t_terminal'], dt = args['dt'])
    
    # step 1. real data loaded
    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index(drop = True)
    df_disruption = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List.csv", encoding='euc-kr').reset_index(drop = True)
    
    # initial state generator
    df, scaler_0D, scaler_ctrl = preparing_initial_dataset(df, cols_0D, cols_control, 'Robust')
    
    init_generator = InitGenerator(df, t_init, cols_0D, cols_control, seq_len, pred_len, True, None)
    
    # Replay Buffer
    memory = ReplayBuffer(capacity=100000)
    
    # Actor and Critic Network
    input_dim = len(cols_0D)
    n_actions = len(cols_control)
    
    # define OU noise
    ou_noise = OUNoise(n_actions, pred_len, mu = 0, theta = 0.15, max_sigma = 0.5, min_sigma = 0.1, decay_period=10000)
    
    # policy and value network
    policy_network = Actor(input_dim, seq_len, pred_len, config.DDPG_CONF['mlp_dim'], n_actions)
    target_policy_network = Actor(input_dim, seq_len, pred_len, config.DDPG_CONF['mlp_dim'], n_actions)
    
    value_network = Critic(input_dim, seq_len, pred_len, config.DDPG_CONF['mlp_dim'], n_actions)
    target_value_network = Critic(input_dim, seq_len, pred_len, config.DDPG_CONF['mlp_dim'], n_actions)
    
    # gpu allocation
    policy_network.to(device)
    target_policy_network.to(device)

    value_network.to(device)
    target_value_network.to(device)
    
    # optimizer
    value_optimizer = torch.optim.AdamW(value_network.parameters(), lr = lr)
    policy_optimizer = torch.optim.AdamW(policy_network.parameters(), lr = lr)

    value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'mean')
    
    # optimization
    print("############### DDPG Training Process ###################")
    save_best = os.path.join("./weights/", "{}_best.pt".format(tag))
    save_last = os.path.join("./weights/", "{}_last.pt".format(tag))
    
    episode_durations, episode_rewards = train_ddpg(
        env, 
        ou_noise,
        init_generator,
        memory,
        policy_network,
        value_network,
        target_policy_network,
        target_value_network,
        policy_optimizer,
        value_optimizer,
        value_loss_fn,
        batch_size,
        gamma,
        device,
        min_value,
        max_value,
        tau,
        num_episode,
        verbose,
        save_best,
        save_last
    )
    
    plt.subplot(1,2,1)
    plt.plot(range(1, num_episode + 1), episode_durations, 'r--', label = 'episode duration')
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, num_episode + 1), episode_rewards, 'b--', label = 'episode reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.savefig("./result/DDPG_episode_reward.png")