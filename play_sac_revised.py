from src.rl.env import NeuralEnv
from src.nn_env.transformer import Transformer
from src.nn_env.forgetting import DFwrapper
from src.rl.rewards import RewardSender
from src.rl.utility import InitGenerator, preparing_initial_dataset, get_range_of_output, plot_virtual_operation
from src.rl.sac import GaussianPolicy, evaluate_sac
from src.rl.buffer import ReplayBuffer
from src.rl.actions import NormalizedActions, ClippingActions
from src.config import Config
from src.rl.video_generator import generate_control_performance
import torch
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="Playing SAC algorithms for tokamak plasma control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "SAC_diff")
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # scenario for training
    parser.add_argument("--shot_num", type = int, default = 21747)
    parser.add_argument("--shot_random", type = bool, default = False)
    parser.add_argument("--t_init", type = float, default = 0.0)
    parser.add_argument("--t_terminal", type = float, default = 10.0)
    parser.add_argument("--dt", type = float, default = 0.05)
    
    # predictor config
    parser.add_argument("--predictor_weight", type = str, default = "./weights/Transformer_seq10_dis1_best.pt")
    parser.add_argument("--use_DF", type = bool, default = False)
    parser.add_argument('--scale_DF', type = float, default = 0.1)
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

    if args['use_DF']:
        model = DFwrapper(model, args['scale_DF'])

    model.to(device)
    model.load_state_dict(torch.load(args['predictor_weight']))

    # reward 
    targets_dict = config.DEFAULT_TARGETS

    # reward
    reward_sender = RewardSender(targets_dict, total_cols = cols_0D)
    
    # step 1. real data loaded
    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index(drop = True)
    df_disruption = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List.csv", encoding='euc-kr').reset_index(drop = True)
    
    # initial state generator
    df, scaler_0D, scaler_ctrl = preparing_initial_dataset(df, cols_0D, cols_control, 'Robust')
    
    init_generator = InitGenerator(df, t_init, cols_0D, cols_control, seq_len, pred_len, args['shot_random'], None)
    
    # info for range of action space
    range_info = get_range_of_output(df, cols_control)
    rate_range_info = config.CTRL_DIFF_RANGE
    
    # environment
    env = NeuralEnv(predictor=model, device = device, reward_sender = reward_sender, seq_len = seq_len, pred_len = pred_len, range_info = range_info, t_terminal = args['t_terminal'], dt = args['dt'], cols_control=config.DEFAULT_COLS_CTRL, limit_ctrl_rate=True, rate_range_info=rate_range_info)
    
    # action rapper
    # env = NormalizedActions(env)
    env = ClippingActions(env)
    
    # Actor and Critic Network
    input_dim = len(cols_0D)
    n_actions = len(cols_control)
    
    # policy and value network
    policy_network = GaussianPolicy(input_dim, seq_len, pred_len, config.SAC_CONF['mlp_dim'], n_actions)
    
    # gpu allocation
    policy_network.to(device)
    
    # load best weight
    save_best = os.path.join("./weights/", "{}_last.pt".format(tag))
    policy_network.load_state_dict(torch.load(save_best))
    
    # Tokamak plasma operation by DDPG algorithm
    shot_num = args['shot_num']
    
    # load real shot information
    env.load_shot_info(df[df.shot == shot_num].copy(deep = True))
    
    print("############### Tokamak plasma operation by SAC algorithm ###################")
    
    print("target parameter : {}".format(list(targets_dict.keys())))
    print("plasma state : {}".format(cols_0D))
    print("control value : {}".format(cols_control))
    
    state_list, action_list, reward_list = evaluate_sac(
        env, 
        init_generator,
        policy_network,
        device,
        shot_num
    )
    
    # plot operation simulation
    total_state, total_action = plot_virtual_operation(
        env,
        state_list,
        action_list,
        reward_list,
        seq_len,
        pred_len,
        shot_num,
        targets_dict,
        config.COL2STR,
        scaler_0D,
        scaler_ctrl,
        tag,
        save_dir
    )
    
    # gif file generation
    title = "{}_ani_shot_{}_operation_control".format(args['tag'], shot_num)
    save_file = os.path.join(save_dir, "{}.gif".format(title))
    generate_control_performance(
        save_file,
        total_state,
        total_action,
        cols_0D,
        cols_control,
        targets_dict,
        "{}_shot_{}_operation_control".format(args['tag'], shot_num),
        args['dt'],
        24,
    )