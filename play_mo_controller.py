from src.rl.env import NeuralEnv
from src.nn_env.transformer import Transformer
from src.nn_env.NStransformer import NStransformer
from src.nn_env.SCINet import SimpleSCINet
from src.nn_env.forgetting import DFwrapper
from src.rl.rewards import RewardSender
from src.rl.utility import InitGenerator, preparing_initial_dataset, get_range_of_output, plot_virtual_operation
from src.rl.sac import GaussianPolicy, evaluate_sac
from src.rl.actions import NormalizedActions, ClippingActions
from src.rl.video_generator import generate_control_performance
from src.GSsolver.model import PINN, ContourRegressor
from src.config import Config
import torch
import numpy as np
import argparse, os
import pandas as pd
import warnings

warnings.filterwarnings(action = 'ignore')

# temporal info
policy_set = [
    './weights/MORL/GPI-LS_multi-objective_Transformer_17_best.pt', 
    './weights/MORL/GPI-LS_multi-objective_Transformer_33_best.pt', 
    './weights/MORL/GPI-LS_multi-objective_Transformer_61_best.pt'
]

weight_support = [
    [1., 0.], 
    [0., 1.], 
    [0.5959, 0.4041]
]

def parsing():
    parser = argparse.ArgumentParser(description="Playing MORL algorithms for tokamak plasma control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "")
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # scenario for playing
    parser.add_argument("--shot_num", type = int, default = 30399)
    parser.add_argument("--shot_random", type = bool, default = False)
    parser.add_argument("--t_init", type = float, default = 0.0)
    parser.add_argument("--t_terminal", type = float, default = 3.0)
    parser.add_argument("--dt", type = float, default = 0.05)
    
    # buffer setup: not used as an training purpose
    parser.add_argument("--use_PER", type = bool, default = False)
    
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
    parser.add_argument("--use_CAPS", type = bool, default=False)
    
    # predictor config
    parser.add_argument("--predictor_model", type = str, default = 'Transformer', choices=['Transformer', 'SCINet', 'NStransformer'])
    parser.add_argument("--predictor_weight", type = str, default = "./weights/Transformer_seq_10_pred_1_interval_3_multi-objective_Robust_best.pt")
    parser.add_argument("--contour_regressor_weight", type = str, default = "./weights/contour_best.pt")
    parser.add_argument("--PINN_weight", type = str, default = "./weights/PINN_best.pt")
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
    
    tag = "SAC_multi-objective_{}".format(args['predictor_model'])
    save_dir = args['save_dir']
    seq_len = args['seq_len']
    pred_len = args['pred_len']
    t_init = args['t_init']

    # tag correction
    if args['use_PER']:
        tag = "{}_PER".format(tag)
        
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
        
    config = Config()
    
    # columns for predictor
    cols_0D = config.input_params['multi-objective']['state']
    cols_control = config.input_params['multi-objective']['control']
    
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
        tag = "{}_DF".format(tag)

    model.to(device)
    model.load_state_dict(torch.load(args['predictor_weight']))

    # reward 
    targets_dict = config.control_config['target']['multi-objective']

    # reward
    reward_sender = RewardSender(targets_dict, total_cols = cols_0D)
    
    # step 1. load dataset
    print("=============== Load dataset ===============")
    df = pd.read_csv("./dataset/KSTAR_rl_control_ts_data_extend.csv").reset_index()
    df_disruption = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_2022.csv", encoding='euc-kr').reset_index()
    
    # initial state generator
    df, scaler_0D, scaler_ctrl = preparing_initial_dataset(df, cols_0D, cols_control, 'Robust')
    
    init_generator = InitGenerator(df, t_init, cols_0D, cols_control, seq_len, pred_len, args['shot_random'], None)
    
    # info for output range
    range_info = get_range_of_output(df, cols_control)

    sample_data = np.load("./src/GSsolver/toy_dataset/g028911_004060.npz")
    R = sample_data['R']
    Z = sample_data['Z']
    
    shape_predictor = PINN(
        R = R,
        Z = Z,
        Rc = config.model_config['GS-solver']['Rc'],
        params_dim = config.model_config['GS-solver']['params_dim'],
        n_PFCs = config.model_config['GS-solver']['n_PFCs'],
        hidden_dim = config.model_config['GS-solver']['hidden_dim'],
        alpha_m = config.model_config['GS-solver']['alpha_m'],
        alpha_n = config.model_config['GS-solver']['alpha_n'],
        beta_m = config.model_config['GS-solver']['beta_m'],
        beta_n = config.model_config['GS-solver']['beta_n'],
        lamda = config.model_config['GS-solver']['lamda'],
        beta = config.model_config['GS-solver']['beta'],
        nx = config.model_config['GS-solver']['nx'],
        ny= config.model_config['GS-solver']['ny']
    )
    
    shape_predictor.eval()
    shape_predictor.to(device)
    shape_predictor.load_state_dict(torch.load(args['PINN_weight']))
    
    contour_regressor = ContourRegressor(65, 65, config.model_config['GS-solver']['params_dim'], config.model_config['GS-solver']['n_PFCs'])
    contour_regressor.eval()
    contour_regressor.to(device)
    contour_regressor.load_state_dict(torch.load(args['contour_regressor_weight']))

    # environment
    if args['stochastic']:
        tag = "{}_stochastic".format(tag)
        
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
        shape_predictor = shape_predictor,
        contour_regressor = contour_regressor,
        objective = 'multi-objective',
        scaler_0D = scaler_0D,
        scaler_ctrl = scaler_ctrl,
        use_stochastic=args['stochastic'],
        noise_mean_0D=args['env_noise_mean_0D'], noise_mean_ctrl=args['env_noise_mean_ctrl'],
        noise_std_0D=args['env_noise_std_0D'], noise_std_ctrl=args['env_noise_std_ctrl'],
        noise_scale_0D=args['env_noise_scale_0D'], noise_scale_ctrl=args['env_noise_scale_ctrl']
    )
    
    # action rapper
    if args['use_normalized_action']:
        env = NormalizedActions(env)
        tag = "{}_normalized".format(tag)
        
    if args['use_clip_action']:
        env = ClippingActions(env)
        tag = "{}_clipping".format(tag)
        
    if args['use_CAPS']:
        tag = "{}_CAPS".format(tag)
    
    # policy and critic network
    input_dim = len(cols_0D)
    n_actions = len(cols_control)
    
    if len(args['tag']) > 0:
        tag = "{}_{}".format(tag, args['tag'])
        
    # Tokamak plasma operation shot selection
    shot_num = args['shot_num']
    
    # load real shot information as an initial condition
    env.load_shot_info(df[df.shot == shot_num].copy(deep = True))
    
    print("=========== MORL algorithm control process ===========")
    print("Target parameters for control: {}".format(list(targets_dict.keys())))
    
    # Playing RL algorithm
    policy_network = GaussianPolicy(input_dim, seq_len, pred_len, config.control_config['SAC']['mlp_dim'], n_actions)
    
    # gpu allocation
    policy_network.to(device)
    
    # load best model
    save_best = policy_set[1]
    policy_network.load_state_dict(torch.load(save_best))

    state_list, action_list, reward_list = evaluate_sac(
        env, 
        init_generator,
        policy_network,
        device,
        shot_num
    )
    
    # Visualization process
    print("=============== Visualization process ===============")
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
        save_dir,
        config.input_params['visualization']
    )
    
    # gif file generation
    title = "{}_ani_shot_{}_operation_control".format(tag, shot_num)
    save_file = os.path.join(save_dir, "{}.gif".format(title))
    generate_control_performance(
        save_file,
        total_state,
        env,
        cols_0D,
        targets_dict,
        title,
        args['dt'],
        12,
        env.seq_len, 
        True,
        config.input_params['visualization']
    )
     