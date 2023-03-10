from src.rl.evaluate import evaluate_ddpg
from src.rl.env import NeuralEnv
from src.nn_env.transformer import Transformer
from src.rl.rewards import RewardSender
from src.rl.utility import InitGenerator, preparing_initial_dataset
from src.rl.ddpg import Actor, Critic
from src.rl.buffer import ReplayBuffer
from src.config import Config
import torch
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parsing():
    parser = argparse.ArgumentParser(description="Playing ddpg algorithms for tokamak plasma control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "DDPG")
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # scenario for training
    parser.add_argument("--shot_random", type = bool, default = False)
    parser.add_argument("--t_init", type = float, default = 0.0)
    parser.add_argument("--t_terminal", type = float, default = 16.0)
    parser.add_argument("--dt", type = float, default = 0.05)
        
    # predictor config
    parser.add_argument("--predictor_weight", type = str, default = "./weights/Transformer_seq10_dis1_best.pt")
    parser.add_argument("--seq_len", type = int, default = 10)
    parser.add_argument("--pred_len", type = int, default = 1)
    
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
    seq_len = args['seq_len']
    pred_len = args['pred_len']
    t_init = args['t_init']
        
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
        
    config = Config()
    
    # 0D parameter
    cols_0D = config.DEFAULT_COLS_0D
    
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
    
    # Actor and Critic Network
    input_dim = len(cols_0D)
    n_actions = len(cols_control)
    
    # policy and value network
    policy_network = Actor(input_dim, seq_len, pred_len, config.DDPG_CONF['mlp_dim'], n_actions)
    value_network = Critic(input_dim, seq_len, pred_len, config.DDPG_CONF['mlp_dim'], n_actions)
    
    # gpu allocation
    policy_network.to(device)
    value_network.to(device)
    
    # load best weight
    save_best = os.path.join("./weights/", "{}_last.pt".format(tag))
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
            total_state = state.reshape(seq_len,-1)
            total_action = action.reshape(seq_len + pred_len,-1)
        else:
            total_state = np.concatenate((total_state, state[-pred_len:,:].reshape(pred_len,-1)), axis= 0)
            total_action = np.concatenate((total_action, action[:,:].reshape(pred_len,-1)), axis= 0)

    # re-scaling : considering the physical unit and scale of the system
    total_state = scaler_0D.inverse_transform(total_state)
    total_action = scaler_ctrl.inverse_transform(total_action)

    # 0D parameter plot
    title = "shot_{}_operation_0D".format(shot_num)
    save_file = os.path.join(save_dir, "{}.png".format(title))
    fig, axes = plt.subplots(len(cols_0D), 1, figsize = (16,12), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    for i, (ax, col) in enumerate(zip(axes.ravel(), cols_0D)):
        if col in list(targets_dict.keys()):
            hist = total_state[:,i]
            ax.plot(hist, 'k', label = "{}-NN".format(col))
            ax.axhline(targets_dict[col], xmin = 0, xmax = 1)
            ax.set_ylabel(config.COL2STR[col])
            ax.legend(loc = "upper right")
        else:
            hist = total_state[:,i]
            ax.plot(hist, 'k', label = "{}-NN".format(col))
            ax.set_ylabel(config.COL2STR[col])
            ax.legend(loc = "upper right")
            
    ax.set_xlabel('time')
    fig.tight_layout()
    plt.savefig(save_file)
    
    # control value plot
    title = "shot_{}_operation_control".format(shot_num)
    save_file = os.path.join(save_dir, "{}.png".format(title))
    fig, axes = plt.subplots(len(cols_control)//2, 2, figsize = (16,10), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    for i, (ax, col) in enumerate(zip(axes.ravel(), cols_control)):
        hist = total_action[:,i]
        ax.plot(hist, 'k', label = "{}".format(col))
        ax.set_ylabel(config.COL2STR[col])
        ax.legend(loc = "upper right")
            
    ax.set_xlabel('time')
    fig.tight_layout()
    plt.savefig(save_file)