from src.rl.env import NeuralEnv
from src.nn_env.transformer import Transformer
from src.nn_env.NStransformer import NStransformer
from src.nn_env.SCINet import SimpleSCINet
from src.nn_env.forgetting import DFwrapper
from src.rl.rewards import RewardSender
from src.rl.utility import InitGenerator, preparing_initial_dataset, get_range_of_output, plot_rl_status
from src.rl.sac import GaussianPolicy, TwinnedQNetwork
from src.rl.buffer import ReplayBuffer
from src.rl.PER import PER
from src.rl.actions import NormalizedActions, ClippingActions
from src.config import Config
from src.morl.utility import plot_pareto_front
import torch
import argparse, os
import pandas as pd
import warnings

warnings.filterwarnings(action = 'ignore')

policy_set = [
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_40_5_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_40_44_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_40_52_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_41_16_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_41_38_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_41_50_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_42_21_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_42_22_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_42_31_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_42_52_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_42_54_best.pt",   
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_43_2_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_43_8_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_43_28_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_43_36_best.pt",
    "./weights/MORL/GPI-LS_multi-objective_Transformer_seed_43_53_best.pt",
    './weights/MORL/GPI-LS_multi-objective_Transformer_seed_44_8_best.pt', 
    './weights/MORL/GPI-LS_multi-objective_Transformer_seed_44_26_best.pt', 
    './weights/MORL/GPI-LS_multi-objective_Transformer_seed_44_34_best.pt', 
    './weights/MORL/GPI-LS_multi-objective_Transformer_seed_44_47_best.pt',
]

weight_support = [
    [0, 1],
    [0.3744, 0.6256],
    [0., 1.],
    [0.3585, 0.6415],
    [0.6637, 0.3363],
    [0.7590, 0.2410],
    [0.6897, 0.3103],
    [0.7691, 0.2309],
    [0.7691, 0.2309],
    [0.3853, 0.6147],
    [0.3853, 0.6147],
    [0., 1.],
    [0.6082, 0.3918],
    [0.2197, 0.7803],
    [0.2360, 0.7640],
    [0.1425, 0.8575],
    [1., 0.], 
    [0.1820, 0.8180],
    [0.2551, 0.7449],
    [0.1850, 0.8150],
]

def parsing():
    parser = argparse.ArgumentParser(description="Training MORL algorithm (GPI-LS) for tokamak plasma control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "")
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 1)
    
    # scenario for training
    parser.add_argument("--shot_random", type = bool, default = True)
    parser.add_argument("--t_init", type = float, default = 0.0)
    parser.add_argument("--t_terminal", type = float, default = 10.0)
    parser.add_argument("--dt", type = float, default = 0.05)
    
    # environment setup
    parser.add_argument("--stochastic", type = bool, default = False)
    parser.add_argument("--use_normalized_action", type = bool, default = False)
    parser.add_argument("--use_clip_action", type = bool, default=False)
    parser.add_argument("--env_noise_scale_0D", type = float, default = 0.1)
    parser.add_argument("--env_noise_scale_ctrl", type = float, default = 0.1)
    parser.add_argument("--env_noise_mean_0D", type = float, default = 0)
    parser.add_argument("--env_noise_mean_ctrl", type = float, default = 0)
    parser.add_argument("--env_noise_std_0D", type = float, default = 1.0)
    parser.add_argument("--env_noise_std_ctrl", type = float, default = 1.0)
    
    # predictor config
    parser.add_argument("--predictor_model", type = str, default = 'Transformer', choices=['Transformer', 'SCINet', 'NStransformer'])
    parser.add_argument("--predictor_weight", type = str, default = "./weights/Transformer_seq_10_pred_1_interval_3_multi-objective_Robust_best.pt")
    parser.add_argument("--use_DF", type = bool, default = False)
    parser.add_argument('--scale_DF', type = float, default = 0.1)
    parser.add_argument("--seq_len", type = int, default = 10)
    parser.add_argument("--pred_len", type = int, default = 1)
    
    args = vars(parser.parse_args())

    return args

# torch device state
print("=============== Device setup ===============")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()
    
if __name__ == "__main__":
    
    # parsing
    args = parsing()
    
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
    
    # columns for predictor
    cols_0D = config.input_params["multi-objective"]['state']
    cols_control = config.input_params["multi-objective"]['control']
    
    # number of objectives
    num_objective = len(config.control_config['target']['multi-objective'].keys())
    
    # predictor
    if args['predictor_model'] == 'Transformer':
        model = Transformer(
            n_layers = config.model_config[args['predictor_model']]['n_layers'], 
            n_heads = config.model_config[args['predictor_model']]['n_heads'], 
            dim_feedforward = config.model_config[args['predictor_model']]['dim_feedforward'], 
            dropout = config.model_config[args['predictor_model']]['dropout'],        
            RIN = config.model_config[args['predictor_model']]['RIN'],
            input_0D_dim = len(cols_0D),
            input_0D_seq_len = seq_len,
            input_ctrl_dim = len(cols_control),
            input_ctrl_seq_len = seq_len + pred_len,
            output_0D_pred_len = pred_len,
            output_0D_dim = len(cols_0D),
            feature_dim = config.model_config[args['predictor_model']]['feature_0D_dim'],
            noise_mean = config.model_config[args['predictor_model']]['noise_mean'],
            noise_std = config.model_config[args['predictor_model']]['noise_std'],
            kernel_size = config.model_config[args['predictor_model']]['kernel_size']
        )
        
    elif args['predictor_model'] == 'SCINet':
        model = NStransformer(
            n_layers = config.model_config[args['predictor_model']]['n_layers'], 
            n_heads = config.model_config[args['predictor_model']]['n_heads'], 
            dim_feedforward = config.model_config[args['predictor_model']]['dim_feedforward'], 
            dropout = config.model_config[args['predictor_model']]['dropout'],        
            input_0D_dim = len(cols_0D),
            input_0D_seq_len = seq_len,
            input_ctrl_dim = len(cols_control),
            input_ctrl_seq_len = seq_len + pred_len,
            output_0D_pred_len = pred_len,
            output_0D_dim = len(cols_0D),
            feature_0D_dim = config.model_config[args['predictor_model']]['feature_0D_dim'],
            feature_ctrl_dim = config.model_config[args['predictor_model']]['feature_ctrl_dim'],
            noise_mean = config.model_config[args['predictor_model']]['noise_mean'],
            noise_std = config.model_config[args['predictor_model']]['noise_std'],
            kernel_size = config.model_config[args['predictor_model']]['kernel_size']
        )
        
    elif args['predictor_model'] == 'NStransformer':
        model = SimpleSCINet(
            output_len = pred_len,
            input_len = seq_len,
            output_dim = len(cols_0D),
            input_0D_dim = len(cols_0D),
            input_ctrl_dim = len(cols_control),        
            hid_size = config.model_config[args['predictor_model']]['hid_size'],
            num_levels = config.model_config[args['predictor_model']]['num_levels'],
            num_decoder_layer = config.model_config[args['predictor_model']]['num_decoder_layer'],
            concat_len = config.model_config[args['predictor_model']]['concat_len'],
            groups = config.model_config[args['predictor_model']]['groups'],
            kernel = config.model_config[args['predictor_model']]['kernel'],
            dropout = config.model_config[args['predictor_model']]['dropout'],
            single_step_output_One = config.model_config[args['predictor_model']]['single_step_output_One'],
            positionalE = config.model_config[args['predictor_model']]['positionalE'],
            modified = config.model_config[args['predictor_model']]['modified'],
            RIN = config.model_config[args['predictor_model']]['RIN'],
            noise_mean = config.model_config[args['predictor_model']]['noise_mean'],
            noise_std = config.model_config[args['predictor_model']]['noise_std']
        )
    
    if args['use_DF']:
        model = DFwrapper(model, args['scale_DF'])

    model.to(device)
    model.load_state_dict(torch.load(args['predictor_weight']))

    # reward 
    targets_dict = config.control_config['target']["multi-objective"]

    # reward
    reward_sender = RewardSender(targets_dict, total_cols = cols_0D)
    
    # step 1. load dataset
    print("=============== Load dataset ===============")
    df = pd.read_csv("./dataset/KSTAR_rl_control_ts_data_extend.csv").reset_index()
    df_disruption = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_2022.csv", encoding='euc-kr').reset_index()
    
    # initial state generator
    df, scaler_0D, scaler_ctrl = preparing_initial_dataset(df, cols_0D, cols_control, 'Robust')
    
    init_generator = InitGenerator(df, t_init, cols_0D, cols_control, seq_len, pred_len, True, None)
    
    # info for output range
    range_info = get_range_of_output(df, cols_control)
         
    env = NeuralEnv(
        predictor=model, 
        device = device, 
        reward_sender = reward_sender, 
        seq_len = seq_len, 
        pred_len = pred_len, 
        range_info = range_info, 
        t_terminal = args['t_terminal'], 
        dt = args['dt'], 
        cols_control=cols_control,
        objective = "multi-objective",
        use_stochastic=args['stochastic'],
        noise_mean_0D=args['env_noise_mean_0D'], noise_mean_ctrl=args['env_noise_mean_ctrl'],
        noise_std_0D=args['env_noise_std_0D'], noise_std_ctrl=args['env_noise_std_ctrl'],
        noise_scale_0D=args['env_noise_scale_0D'], noise_scale_ctrl=args['env_noise_scale_ctrl'],
        gamma = 0.95
    )
        
    # action rapper
    if args['use_normalized_action']:
        env = NormalizedActions(env)
        
    if args['use_clip_action']:
        env = ClippingActions(env)

    # policy and critic network
    input_dim = len(cols_0D)
    n_actions = len(cols_control)
    
    # Load policy network
    policy_network = GaussianPolicy(input_dim, seq_len, pred_len, config.control_config['SAC']['mlp_dim'], n_actions)
   
    # gpu allocation
    policy_network.to(device)
    
    plot_pareto_front(
        init_generator,
        env,
        policy_network,
        policy_set,
        weight_support,
        ['\\betan', '\\kappa'],
        list(config.control_config['target']["multi-objective"].keys()),
        device,
        "./result/MORL_GPI_LS_Pareto_frontier_total.png",
        "./weights/SAC_shape-control_Transformer_best.pt",
        16,
        50
    )