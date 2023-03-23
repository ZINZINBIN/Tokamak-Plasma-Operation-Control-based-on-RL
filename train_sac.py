from src.rl.env import NeuralEnv
from src.nn_env.transformer import Transformer
from src.nn_env.forgetting import DFwrapper
from src.rl.rewards import RewardSender
from src.rl.utility import InitGenerator, preparing_initial_dataset, get_range_of_output, plot_rl_status
from src.rl.sac import GaussianPolicy, TwinnedQNetwork, train_sac
from src.rl.buffer import ReplayBuffer
from src.rl.actions import NormalizedActions
from src.config import Config
import torch
import argparse, os
import pandas as pd
import warnings

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="training sac algorithms for tokamak plasma control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "SAC")
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # scenario for training
    parser.add_argument("--shot_random", type = bool, default = True)
    parser.add_argument("--t_init", type = float, default = 0.0)
    parser.add_argument("--t_terminal", type = float, default = 10.0)
    parser.add_argument("--dt", type = float, default = 0.05)
    
    # DDPG training setup
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--num_episode", type = int, default = 2048)  
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--gamma", type = float, default = 0.995)
    parser.add_argument("--min_value", type = float, default = -10.0)
    parser.add_argument("--max_value", type = float, default = 10.0)
    parser.add_argument("--tau", type = float, default = 0.01)
    parser.add_argument("--verbose", type = int, default = 4)
    
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
    
    init_generator = InitGenerator(df, t_init, cols_0D, cols_control, seq_len, pred_len, True, None)
    
    # info for output range
    range_info = get_range_of_output(df, cols_control)
    
    # environment
    env = NeuralEnv(predictor=model, device = device, reward_sender = reward_sender, seq_len = seq_len, pred_len = pred_len, range_info = range_info, t_terminal = args['t_terminal'], dt = args['dt'])
    
    # action rapper
    # env = NormalizedActions(env)
    
    # Replay Buffer
    memory = ReplayBuffer(capacity=1000000)
    
    # policy and critic network
    input_dim = len(cols_0D)
    n_actions = len(cols_control)
    
    value_network = TwinnedQNetwork(input_dim, seq_len, pred_len, config.SAC_CONF['mlp_dim'], n_actions)
    policy_network = GaussianPolicy(input_dim, seq_len, pred_len, config.SAC_CONF['mlp_dim'], n_actions)
    target_value_network = TwinnedQNetwork(input_dim, seq_len, pred_len, config.SAC_CONF['mlp_dim'], n_actions)
    
    # temperature parameter
    log_alpha = torch.zeros(1, requires_grad=True)
    target_entropy = -torch.prod(torch.Tensor((n_actions,)))
    
    # gpu allocation
    policy_network.to(device)
    value_network.to(device)
    target_value_network.to(device)
    
    # optimizer
    q1_optimizer = torch.optim.AdamW(value_network.Q1.parameters(), lr = lr)
    q2_optimizer = torch.optim.AdamW(value_network.Q2.parameters(), lr = lr)
    policy_optimizer = torch.optim.AdamW(policy_network.parameters(), lr = lr)
    alpha_optimizer = torch.optim.AdamW([log_alpha], lr = lr)
    
    # loss function for critic network
    value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'mean')
    
    # optimization
    print("############### SAC Training Process ###################")
    save_best = os.path.join("./weights/", "{}_best.pt".format(tag))
    save_last = os.path.join("./weights/", "{}_last.pt".format(tag))
    
    target_value_result, episode_reward = train_sac(
        env, 
        init_generator,
        memory,
        policy_network,
        value_network,
        target_value_network,
        target_entropy,
        log_alpha, 
        policy_optimizer,
        q1_optimizer,
        q2_optimizer,
        alpha_optimizer,
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
        save_last,
        scaler_0D
    )
    
    plot_rl_status(target_value_result, episode_reward, tag, config.COL2STR, "./result/SAC_episode_reward.png")