from src.rl.env import NeuralEnv
from src.nn_env.transformer import Transformer
from src.nn_env.NStransformer import NStransformer
from src.nn_env.SCINet import SimpleSCINet
from src.nn_env.forgetting import DFwrapper
from src.rl.rewards import RewardSender
from src.rl.utility import InitGenerator, preparing_initial_dataset, get_range_of_output, plot_rl_status
from src.rl.sac import GaussianPolicy, TwinnedQNetwork
from src.morl.sac_gpi_ls import train_sac_linear_support
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

def parsing():
    parser = argparse.ArgumentParser(description="Training MORL algorithm (GPI-LS) for tokamak plasma control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "")
    parser.add_argument("--save_dir", type = str, default = "./result")
    parser.add_argument("--seed", type = int, default = 42)
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 1)
    
    # scenario for training
    parser.add_argument("--shot_random", type = bool, default = True)
    parser.add_argument("--t_init", type = float, default = 0.0)
    parser.add_argument("--t_terminal", type = float, default = 10.0)
    parser.add_argument("--dt", type = float, default = 0.05)
    
    # ReplayBuffer setting
    parser.add_argument("--capacity", type = int, default = 10000)
    parser.add_argument("--use_PER", type = bool, default = False)
    
    # training setup
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--num_episode", type = int, default = 512)  
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--gamma", type = float, default = 0.995)
    parser.add_argument("--min_value", type = float, default = -10.0)
    parser.add_argument("--max_value", type = float, default = 10.0)
    parser.add_argument("--tau", type = float, default = 0.01)
    parser.add_argument("--verbose", type = int, default = 1)
    parser.add_argument("--verbose_policy",type=int, default = 32)
    parser.add_argument("--use_CAPS", type = bool, default=False)
    parser.add_argument("--lamda_temporal_smoothness", type = float, default = 4.0)
    parser.add_argument("--lamda_spatial_smoothness", type = float, default = 1.0)
    parser.add_argument("--max_gpi_ls_iters", type = int, default = 64)
    
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
    
    tag = "GPI-LS_multi-objective_{}".format(args['predictor_model'])
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
        tag = "{}_DF".format(tag)
        
    # seed
    tag = "{}_seed_{}".format(tag, args['seed'])

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
        objective = "multi-objective",
        use_stochastic=args['stochastic'],
        noise_mean_0D=args['env_noise_mean_0D'], noise_mean_ctrl=args['env_noise_mean_ctrl'],
        noise_std_0D=args['env_noise_std_0D'], noise_std_ctrl=args['env_noise_std_ctrl'],
        noise_scale_0D=args['env_noise_scale_0D'], noise_scale_ctrl=args['env_noise_scale_ctrl'],
        gamma = gamma
    )
        
    # action rapper
    if args['use_normalized_action']:
        env = NormalizedActions(env)
        tag = "{}_normalized".format(tag)
        
    if args['use_clip_action']:
        env = ClippingActions(env)
        tag = "{}_clipping".format(tag)
    
    # Replay Buffer
    if args['use_PER']:
        memory = PER(capacity=args['capacity'])
    else:
        memory = ReplayBuffer(capacity=args['capacity'])
    
    # policy and critic network
    input_dim = len(cols_0D)
    n_actions = len(cols_control)
    
    # CAPS: Regularzing Action Policies for smooth control with reinforcement learning
    if args['use_CAPS']:
        tag = "{}_CAPS".format(tag)
        lamda_temporal_smoothness = args['lamda_temporal_smoothness']
        lamda_spatial_smoothness = args['lamda_spatial_smoothness']
    else:
        lamda_temporal_smoothness = 0
        lamda_spatial_smoothness = 0
    
    if len(args['tag']) > 0:
        tag = "{}_{}".format(tag, args['tag'])
    
    # Training RL algorithm
    value_network = TwinnedQNetwork(input_dim, seq_len, pred_len, config.control_config['SAC']['mlp_dim'], n_actions)
    policy_network = GaussianPolicy(input_dim, seq_len, pred_len, config.control_config['SAC']['mlp_dim'], n_actions)
    target_value_network = TwinnedQNetwork(input_dim, seq_len, pred_len, config.control_config['SAC']['mlp_dim'], n_actions)
    
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
    if args['use_PER']:
        value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'none')
    else:
        value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'mean')
    
    # optimization
    print("=========== GPI-LS: SAC version MORL algorithm process ===========")
    save_dir = os.path.join("./weights/", "MORL")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("Process : {}".format(tag))
    
    weight_support, ccs, policy_set = train_sac_linear_support(
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
        args['verbose_policy'],
        save_dir,
        scaler_0D,
        args['use_CAPS'],
        lamda_temporal_smoothness,
        lamda_spatial_smoothness,
        args['max_gpi_ls_iters'],
        num_objective,
        tag,
        args['seed']
    )
    
    print("# of weights for Pareto-optimal : ", len(weight_support))
    print("# of CCS corresponding the weights : ", len(ccs))
    print("# of policy corresponding the weights : ", len(policy_set))
  
    print("weight support: ", weight_support)
    print("policy set: ", policy_set)
    
    result = pd.DataFrame({})
    result['weight-support'] = weight_support
    result['CCS'] = ccs
    result['policy'] = policy_set
    result.to_csv("./result/MORL-GPI-LS_{}.csv".format(tag))
    
    plot_pareto_front(
        init_generator,
        env,
        policy_network,
        policy_set,
        weight_support,
        ['\\betan', '\\kappa'],
        list(config.control_config['target']["multi-objective"].keys()),
        device,
        "./result/MORL_GPI_LS_Pareto_frontier_{}.png".format(tag),
        "./weights/SAC_shape-control_Transformer_best.pt",
        8,
        32
    )