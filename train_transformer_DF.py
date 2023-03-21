import torch
import argparse
import numpy as np
import pandas as pd
from src.config import Config
from src.nn_env.utility import preparing_0D_dataset, get_range_of_output
from src.nn_env.dataset import DatasetFor0D, DatasetForMultiStepPred
from src.nn_env.transformer import Transformer
from src.nn_env.train import train
from src.nn_env.loss import CustomLoss
from src.nn_env.forgetting import DFwrapper
from src.nn_env.evaluate import evaluate, evaluate_multi_step
from src.nn_env.predict import generate_shot_data_from_real, generate_shot_data_from_self
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="training NN based environment - Transformer with differentiate forgetting")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--lr", type = float, default = 2e-4)
parser.add_argument("--gpu_num", type = int, default = 3)
parser.add_argument("--num_epoch", type = int, default = 128)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--verbose", type = int, default = 4)
parser.add_argument("--max_norm_grad", type = float, default = 1.0)
parser.add_argument("--root_dir", type = str, default = "./weights/")
parser.add_argument("--tag", type = str, default = "Transformer_DF")
parser.add_argument("--use_scaler", type = bool, default = True)
parser.add_argument("--scaler", type = str, default = 'Robust', choices = ['Standard', 'Robust', 'MinMax'])
parser.add_argument("--seq_len", type = int, default = 10)
parser.add_argument("--pred_len", type = int, default = 1)
parser.add_argument("--interval", type = int, default = 3)
parser.add_argument("--scale", type = float, default = 0.1)
parser.add_argument("--multi_step_validation", type = bool, default = False)

args = vars(parser.parse_args())

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:{}".format(args['gpu_num'])
else:
    device = 'cpu'

if __name__ == "__main__":
    
    config = Config()

    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index()
    df_disruption = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List.csv", encoding='euc-kr').reset_index()

    # nan interpolation
    df.interpolate(method = 'linear', limit_direction = 'forward')

    # columns for use
    # 0D parameter
    cols_0D = config.DEFAULT_COLS_0D
    
    # control value / parameter
    cols_control = config.DEFAULT_COLS_CTRL
    
    ts_train, ts_valid, ts_test, scaler_0D, scaler_ctrl = preparing_0D_dataset(df, df_disruption, cols_0D, cols_control, args['scaler'])
    
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
        test_data = DatasetFor0D(ts_test.copy(deep = True), df_disruption, seq_len, seq_len + pred_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    
    print("train data : ", train_data.__len__())
    print("valid data : ", valid_data.__len__())
    print("test data : ", test_data.__len__())

    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = 4, shuffle = True, pin_memory = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = 4, shuffle = True, pin_memory = False)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 4, shuffle = True, pin_memory = True)
    
    # data range
    ts_data = pd.concat([train_data.ts_data, valid_data.ts_data], axis = 1)
    range_info = get_range_of_output(ts_data, cols_0D)
    
    # transformer model argument
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
        range_info = range_info,
        noise_mean = config.TRANSFORMER_CONF['noise_mean'],
        noise_std = config.TRANSFORMER_CONF['noise_std']
    )
    
    model = DFwrapper(model, len(cols_0D), args['scale'])
    model.summary()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=args['gamma'])
    
    import os
    
    if config.TRANSFORMER_CONF['RIN']:
        tag = "{}_seq{}_dis{}_RevIN".format(args['tag'], args['seq_len'], args['pred_len'])
    else:
        tag = "{}_seq{}_dis{}".format(args['tag'], args['seq_len'], args['pred_len'])
    
    save_best_dir = os.path.join(args['root_dir'], "{}_best.pt".format(tag))
    save_last_dir = os.path.join(args['root_dir'], "{}_last.pt".format(tag))
    tensorboard_dir = os.path.join("./runs/", "tensorboard_{}".format(tag))

    # loss_fn = CustomLoss() 
    loss_fn = torch.nn.MSELoss(reduction = 'mean')
    
    print("\n##### training process #####\n")
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
        test_loss, mse, rmse, mae, r2 = evaluate(
            test_loader,
            model,
            optimizer,
            loss_fn,
            device,
        )
    
    shot_num = ts_test.shot.iloc[-1]
    df_shot = ts_test[ts_test.shot == shot_num].reset_index(drop = True)
    
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