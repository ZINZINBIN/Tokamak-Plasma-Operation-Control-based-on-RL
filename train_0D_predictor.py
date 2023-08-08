import torch
import argparse
import numpy as np
import pandas as pd
from src.config import Config
from src.nn_env.utility import preparing_0D_dataset, get_range_of_output
from src.nn_env.dataset import DatasetFor0D, DatasetForMultiStepPred
from src.nn_env.transformer import Transformer
from src.nn_env.NStransformer import NStransformer
from src.nn_env.SCINet import SimpleSCINet
from src.nn_env.train import train
from src.nn_env.loss import CustomLoss
from src.nn_env.forgetting import DFwrapper
from src.nn_env.evaluate import evaluate, evaluate_multi_step
from src.nn_env.predict import generate_shot_data_from_real, generate_shot_data_from_self
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="training NN based environment - 0D predictor")
    
    # tag
    parser.add_argument("--tag", type = str, default = "")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # cpu workers per gpu
    parser.add_argument("--num_workers", type = int, default = 4)
    
    # target
    parser.add_argument("--objective", type = str, default = "params-control", choices = ['params-control', 'shape-control', 'multi-objective'])
    
    # training setup
    parser.add_argument("--batch_size", type = int, default = 1024)
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--num_epoch", type = int, default = 32)
    parser.add_argument("--verbose", type = int, default = 8)
    parser.add_argument("--max_norm_grad", type = float, default = 1.0)
    parser.add_argument("--multi_step_validation", type = bool, default = False)
    parser.add_argument("--evaluate_multi_step", type = bool, default = False)
    
    # test shot num
    parser.add_argument("--test_shot_num", type = int, default = 30399)
    
    # scheduler for training
    parser.add_argument("--gamma", type = float, default = 0.95)
    parser.add_argument("--step_size", type = int, default = 8)
    
    # directory
    parser.add_argument("--root_dir", type = str, default = "./weights/")
    
    # model
    parser.add_argument("--model", type = str, default = "Transformer", choices = ['Transformer', 'SCINet', 'NStransformer'])
    
    # model properties
    parser.add_argument("--seq_len", type = int, default = 10)
    parser.add_argument("--pred_len", type = int, default = 1)
    parser.add_argument("--interval", type = int, default = 3)
    
    # Forgetting setup
    parser.add_argument("--use_forgetting", type = bool, default = False)
    parser.add_argument("--scale_forgetting", type = float, default = 0.1)
    
    # scaling
    parser.add_argument("--use_scaler", type = bool, default = True)
    parser.add_argument("--scaler", type = str, default = 'Robust', choices = ['Standard', 'Robust', 'MinMax'])

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
    
    args = parsing()
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:{}".format(args['gpu_num'])
    else:
        device = 'cpu'
    
    # configuration
    config = Config()

    # load dataset for training
    print("=============== Load dataset ===============")
    df = pd.read_csv("./dataset/KSTAR_rl_control_ts_data_extend.csv").reset_index()
    df_disruption = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_2022.csv", encoding='euc-kr').reset_index()
    
    # columns for use
    cols_0D = config.input_params[args['objective']]['state']
    cols_control = config.input_params[args['objective']]['control']
    
    # load dataset
    ts_train, ts_valid, ts_test, scaler_0D, scaler_ctrl = preparing_0D_dataset(df, cols_0D, cols_control, args['scaler'])
    
    seq_len = args['seq_len']
    pred_len = args['pred_len']
    interval = args['interval']
    batch_size = args['batch_size']
    pred_cols = cols_0D
    
    # while training, only single-step prediction is used
    train_data = DatasetFor0D(ts_train.copy(deep = True), df_disruption, seq_len, seq_len + pred_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    
    # Meanwhile, validation and test process will use both single-step and multi-step prediction
    if args['multi_step_validation']:
        valid_data = DatasetForMultiStepPred(ts_valid.copy(deep = True), df_disruption, seq_len, seq_len + pred_len, seq_len * 4, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    else:
        valid_data = DatasetFor0D(ts_valid.copy(deep = True), df_disruption, seq_len, seq_len + pred_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    
    if args['multi_step_validation']:
        test_data = DatasetForMultiStepPred(ts_test.copy(deep = True), df_disruption, seq_len, seq_len + pred_len, seq_len * 4, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    else:
        if args['evaluate_multi_step']:
            test_data = DatasetForMultiStepPred(ts_test.copy(deep = True), df_disruption, seq_len, seq_len + pred_len, seq_len * 4, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
        else:
            test_data = DatasetFor0D(ts_test.copy(deep = True), df_disruption, seq_len, seq_len + pred_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    
    print("=============== Dataset information ===============")
    print("train data : ", train_data.__len__())
    print("valid data : ", valid_data.__len__())
    print("test data : ", test_data.__len__())

    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = args['num_workers'], shuffle = True, pin_memory = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = args['num_workers'], shuffle = True, pin_memory = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = args['num_workers'], shuffle = True, pin_memory = True)
    
    # data range
    ts_data = pd.concat([train_data.ts_data, valid_data.ts_data, test_data.ts_data], axis = 1)
    range_info = get_range_of_output(ts_data, cols_0D)
    
    # transformer model argument
    if args['model'] == 'Transformer':
        model = Transformer(
            n_layers = config.model_config[args['model']]['n_layers'], 
            n_heads = config.model_config[args['model']]['n_heads'], 
            dim_feedforward = config.model_config[args['model']]['dim_feedforward'], 
            dropout = config.model_config[args['model']]['dropout'],        
            RIN = config.model_config[args['model']]['RIN'],
            input_0D_dim = len(cols_0D),
            input_0D_seq_len = seq_len,
            input_ctrl_dim = len(cols_control),
            input_ctrl_seq_len = seq_len + pred_len,
            output_0D_pred_len = pred_len,
            output_0D_dim = len(cols_0D),
            feature_dim = config.model_config[args['model']]['feature_0D_dim'],
            range_info = range_info,
            noise_mean = config.model_config[args['model']]['noise_mean'],
            noise_std = config.model_config[args['model']]['noise_std'],
            kernel_size = config.model_config[args['model']]['kernel_size']
        )
        
    elif args['model'] == 'NStransformer':
        model = NStransformer(
            n_layers = config.model_config[args['model']]['n_layers'], 
            n_heads = config.model_config[args['model']]['n_heads'], 
            dim_feedforward = config.model_config[args['model']]['dim_feedforward'], 
            dropout = config.model_config[args['model']]['dropout'],        
            input_0D_dim = len(cols_0D),
            input_0D_seq_len = seq_len,
            input_ctrl_dim = len(cols_control),
            input_ctrl_seq_len = seq_len + pred_len,
            output_0D_pred_len = pred_len,
            output_0D_dim = len(cols_0D),
            feature_0D_dim = config.model_config[args['model']]['feature_0D_dim'],
            feature_ctrl_dim = config.model_config[args['model']]['feature_ctrl_dim'],
            range_info = range_info,
            noise_mean = config.model_config[args['model']]['noise_mean'],
            noise_std = config.model_config[args['model']]['noise_std'],
            kernel_size = config.model_config[args['model']]['kernel_size']
        )
        
    elif args['model'] == 'SCINet':
        model = SimpleSCINet(
            output_len = pred_len,
            input_len = seq_len,
            output_dim = len(cols_0D),
            input_0D_dim = len(cols_0D),
            input_ctrl_dim = len(cols_control),        
            hid_size = config.model_config[args['model']]['hid_size'],
            num_levels = config.model_config[args['model']]['num_levels'],
            num_decoder_layer = config.model_config[args['model']]['num_decoder_layer'],
            concat_len = config.model_config[args['model']]['concat_len'],
            groups = config.model_config[args['model']]['groups'],
            kernel = config.model_config[args['model']]['kernel'],
            dropout = config.model_config[args['model']]['dropout'],
            single_step_output_One = config.model_config[args['model']]['single_step_output_One'],
            positionalE = config.model_config[args['model']]['positionalE'],
            modified = config.model_config[args['model']]['modified'],
            RIN = config.model_config[args['model']]['RIN'],
            noise_mean = config.model_config[args['model']]['noise_mean'],
            noise_std = config.model_config[args['model']]['noise_std']
        )
    
    # If using differentiate forgetting, wrapper is called
    if args['use_forgetting']:
        model = DFwrapper(model, args['scale_forgetting'])

    model.summary()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['step_size'], gamma=args['gamma'])
    
    import os
    
    # tag labeling
    tag = "{}_seq_{}_pred_{}_interval_{}_{}".format(args['model'], args['seq_len'], args['pred_len'], args['interval'], args['objective'])

    if config.model_config[args['model']]['RIN']:
        tag = "{}_RevIN".format(tag)
    
    if args['use_forgetting']:
        tag = "{}_DF".format(tag)
    
    if args['use_scaler']:
        tag = "{}_{}".format(tag, args['scaler'])
        
    if args['multi_step_validation']:
        tag = "{}_msv".format(tag)
    
    if len(args['tag']) > 0:
        tag = "{}_{}".format(tag, args['tag'])
    
    save_best_dir = os.path.join(args['root_dir'], "{}_best.pt".format(tag))
    save_last_dir = os.path.join(args['root_dir'], "{}_last.pt".format(tag))
    tensorboard_dir = os.path.join("./runs/", "tensorboard_{}".format(tag))

    loss_fn = torch.nn.MSELoss(reduction = 'mean')
    
    if os.path.exists(save_last_dir):
        pass
        # model.load_state_dict(torch.load(save_last_dir))
    
    print("=============== Training process ===============")
    print("Process : {}".format(tag))
    train_loss, valid_loss = train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        args['num_epoch'],
        args['verbose'],
        save_best = save_best_dir,
        save_last = save_last_dir,
        max_norm_grad = args['max_norm_grad'],
        tensorboard_dir = tensorboard_dir,
        test_for_check_per_epoch = test_loader,
        multi_step_validation = args['multi_step_validation']
    )

    
    model.load_state_dict(torch.load(save_best_dir))
    
    print("=============== Evaluation process ===============")
    # evaluation process
    if args['multi_step_validation']:
        test_loss, mse, rmse, mae, r2 = evaluate_multi_step(
            test_loader,
            model,
            optimizer,
            loss_fn,
            device,
        )
    else:
        if args['evaluate_multi_step']:
            test_loss, mse, rmse, mae, r2 = evaluate_multi_step(
                test_loader,
                model,
                optimizer,
                loss_fn,
                device,
            )
            
        else:
            test_loss, mse, rmse, mae, r2 = evaluate(
                test_loader,
                model,
                optimizer,
                loss_fn,
                device,
            )
            
    
    print("=============== Auto-regressive prediction ===============")
    shot_num = args['test_shot_num']
    df_shot = df[df.shot == shot_num].reset_index(drop = True)
    
    # virtual experiment shot 
    generate_shot_data_from_self(
        model,
        df_shot,
        seq_len,
        seq_len + pred_len,
        pred_len,
        cols_0D,
        cols_control,
        scaler_0D,
        scaler_ctrl,
        device,
        "shot number : {}".format(shot_num),
        save_dir = os.path.join("./result/", "{}_without_real_data.png".format(tag))
    )
    
    print("=============== Feedforward prediction  ===============")
    # feedback from real data
    generate_shot_data_from_real(
        model,
        df_shot,
        seq_len,
        seq_len + pred_len,
        pred_len,
        cols_0D,
        cols_control,
        scaler_0D,
        scaler_ctrl,
        device,
        "shot number : {}".format(shot_num),
        save_dir = os.path.join("./result/", "{}_with_real_data.png".format(tag))
    )