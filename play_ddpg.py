from src.rl.env import NeuralEnv
from src.nn_env.model import TStransformer
from src.rl.rewards import RewardSender
from src.rl.utility import InitGenerator
from src.rl.ddpg import Actor, Critic
from src.rl.buffer import ReplayBuffer
from src.rl.evaluate import evaluate_ddpg
import torch
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

def parsing():
    parser = argparse.ArgumentParser(description="training ddpg algorithms for tokamak plasma control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "DDPG")
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)


    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--num_episode", type = int, default = 256)
    parser.add_argument("--seq_len", type = int, default = 21)
    parser.add_argument("--pred_len", type = int, default = 1)
    
    parser.add_argument("--t_init", type = float, default = 1.5)
    
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
        
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
    
    # 0D parameters
    cols_0D = [
        '\\q0', '\\q95', '\\ipmhd', '\\kappa', 
        '\\tritop', '\\tribot','\\betap','\\betan',
        '\\li', '\\WTOT_DLM03', '\\ne_inter01',
    ]
    
    # else diagnostics
    cols_diag = [
        '\\ne_tci01', '\\ne_tci02', '\\ne_tci03', '\\ne_tci04', '\\ne_tci05',
    ]

    # control value / parameter
    cols_control = [
        '\\nb11_pnb','\\nb12_pnb','\\nb13_pnb',
        '\\RC01', '\\RC02', '\\RC03',
        '\\VCM01', '\\VCM02', '\\VCM03',
        '\\EC2_PWR', '\\EC3_PWR', 
        '\\ECSEC2TZRTN', '\\ECSEC3TZRTN',
        '\\LV01'
    ]

    # predictor
    model = TStransformer(
        n_features = len(cols_0D + cols_control), 
        feature_dims = 128, 
        max_len = seq_len, 
        n_layers = 4, 
        n_heads = 8, 
        dim_feedforward = 512, 
        dropout = 0.25, 
        mlp_dim = 64, 
        pred_len = pred_len,
        output_dim = len(cols_0D)
    )

    model.to(device)
    model.load_state_dict(torch.load("./weights/TStransformer_best.pt"))

    # reward 
    targets_dict = {
        '\\betap' : 3.0,
        # '\\betan' : 4.0,
        '\\q95' : 4.0,
        # '\\q0' : 1.0
    }

    # reward
    reward_sender = RewardSender(targets_dict, total_cols = cols_0D + cols_control)
    
    # environment
    env = NeuralEnv(predictor=model, device = device, reward_sender = reward_sender, seq_len = seq_len, pred_len = pred_len)
    
    # initial state generator
    # step 1. real data loaded
    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index()
    df_disruption = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List.csv", encoding='euc-kr').reset_index()
    
    # nan interpolation
    df.interpolate(method = 'linear', limit_direction = 'forward')
    
    # columns for use
    ts_cols = cols_0D + cols_control

    # float type
    for col in ts_cols:
        df[col] = df[col].astype(np.float32)
        
    df[cols_control] = df[cols_control].fillna(0)

    df[cols_0D] = df[cols_0D].fillna(method = 'ffill')

    scaler = RobustScaler()
    df[ts_cols] = scaler.fit_transform(df[ts_cols].values)

    init_generator = InitGenerator(df, t_init, cols_0D, cols_control, seq_len, True, None)
    
    # Replay Buffer
    memory = ReplayBuffer(capacity=100000)
    
    # Actor Network
    input_dim = len(cols_0D)
    mlp_dim = 64
    n_actions = len(cols_control)
    
    policy_network = Actor(input_dim, seq_len, mlp_dim, n_actions)    
    policy_network.to(device)
    
    save_best = os.path.join("./weights/", "{}_best.pt".format(tag))
    policy_network.load_state_dict(torch.load(save_best))
 
    # Tokamak plasma operation by DDPG algorithm
    shot_num = 21474
    print("############### Tokamak plasma operation by DDPG algorithm ###################")
    print("target parameter : {}".format(list(targets_dict.keys())))
    print("plasma state : {}".format(cols_0D))
    print("control value : {}".format(cols_control))
    state_list, action_list, reward_list = evaluate_ddpg(
        env, 
        init_generator,
        policy_network,
        device,
        shot_num
    )

    total_state = None
    total_action = None
    
    for state, action in zip(state_list, action_list):
        if total_state is None:
            total_state = state[-1,:].reshape(1,-1)
            total_action = action[-1,:].reshape(1,-1)
        else:
            total_state = np.concatenate((total_state, state[-1,:].reshape(1,-1)), axis= 0)
            total_action = np.concatenate((total_action, action[-1,:].reshape(1,-1)), axis= 0)

    total_state = np.concatenate((total_state, total_action), axis = 1)     
    total_state = scaler.inverse_transform(total_state)
    
    total_action = total_state[:,len(cols_0D):]
    total_state = total_state[:,0:len(cols_0D)]
    
    # 0D parameter plot
    title = "shot_{}_operation_0D".format(shot_num)
    save_file = os.path.join(save_dir, "{}.png".format(title))
    fig, axes = plt.subplots(len(cols_0D), figsize = (6,12), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    for i, (ax, col) in enumerate(zip(axes, cols_0D)):
        if col in list(targets_dict.keys()):
            hist = total_state[:,i]
            ax.plot(hist, 'k', label = "{}-NN".format(col))
            ax.axhline(targets_dict[col], xmin = 0, xmax = 1)
            ax.set_ylabel(col)
            ax.legend(loc = "upper right")
        else:
            hist = total_state[:,i]
            ax.plot(hist, 'k', label = "{}-NN".format(col))
            ax.set_ylabel(col)
            ax.legend(loc = "upper right")
            
    ax.set_xlabel('time')
    fig.tight_layout()
    plt.savefig(save_file)
    
    # control value plot
    title = "shot_{}_operation_control".format(shot_num)
    save_file = os.path.join(save_dir, "{}.png".format(title))
    fig, axes = plt.subplots(len(cols_control), figsize = (6,14), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    for i, (ax, col) in enumerate(zip(axes, cols_control)):
        hist = total_action[:,i]
        ax.plot(hist, 'k', label = "{}".format(col))
        ax.set_ylabel(col)
        ax.legend(loc = "upper right")
            
    ax.set_xlabel('time')
    fig.tight_layout()
    plt.savefig(save_file)