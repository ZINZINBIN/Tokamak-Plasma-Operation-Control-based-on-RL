import torch
import argparse
import numpy as np
import pandas as pd
from src.nn_env.utility import preparing_0D_dataset
from src.nn_env.dataset import DatasetFor0D
from src.nn_env.transformer import Transformer
from src.nn_env.train import train
from src.nn_env.loss import CustomLoss
from src.nn_env.evaluate import evaluate
from src.nn_env.predict import generate_shot_data_from_real, generate_shot_data_from_self
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="training NN based environment - Transformer")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--lr", type = float, default = 2e-4)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--num_epoch", type = int, default = 50)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--verbose", type = int, default = 4)
parser.add_argument("--max_norm_grad", type = float, default = 1.0)
parser.add_argument("--root_dir", type = str, default = "./weights/")
parser.add_argument("--tag", type = str, default = "Transformer")
parser.add_argument("--use_scaler", type = bool, default = True)
parser.add_argument("--scaler", type = str, default = 'Robust', choices = ['Standard', 'Robust', 'MinMax'])
parser.add_argument("--seq_len", type = int, default = 16)
parser.add_argument("--pred_len", type = int, default = 4)
parser.add_argument("--interval", type = int, default = 3)

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

    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index()
    df_disruption = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List.csv", encoding='euc-kr').reset_index()

    # nan interpolation
    df.interpolate(method = 'linear', limit_direction = 'forward')

    # columns for use
    # 0D parameter
    cols_0D = [
        '\\q95', '\\ipmhd', '\\kappa', 
        '\\tritop', '\\tribot','\\betap','\\betan',
        '\\li', '\\WTOT_DLM03', '\\ne_inter01',
        '\\TS_NE_CORE_AVG', '\\TS_TE_CORE_AVG'
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
    
    ts_train, ts_valid, ts_test, scaler_0D, scaler_ctrl = preparing_0D_dataset(df, df_disruption, cols_0D, cols_control, args['scaler'])
    
    seq_len = args['seq_len']
    pred_len = args['pred_len']
    interval = args['interval']
    batch_size = args['batch_size']
    
    pred_cols = cols_0D
    
    train_data = DatasetFor0D(ts_train.copy(deep = True), df_disruption, seq_len, seq_len + pred_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    valid_data = DatasetFor0D(ts_valid.copy(deep = True), df_disruption, seq_len, seq_len + pred_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    test_data = DatasetFor0D(ts_test.copy(deep = True), df_disruption, seq_len, seq_len + pred_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl)
    
    print("train data : ", train_data.__len__())
    print("valid data : ", valid_data.__len__())
    print("test data : ", test_data.__len__())

    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = 4, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = 4, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 4, shuffle = True)

    model = Transformer(
        n_layers = 2, 
        n_heads = 8, 
        dim_feedforward = 1024, 
        dropout = 0.1,        
        RIN = True,
        input_0D_dim = len(cols_0D),
        input_0D_seq_len = seq_len,
        input_ctrl_dim = len(cols_control),
        input_ctrl_seq_len = seq_len + pred_len,
        output_0D_pred_len = pred_len,
        output_0D_dim = len(cols_0D),
        feature_0D_dim = 128,
        feature_ctrl_dim = 128,
    )

    model.summary()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=args['gamma'])
    
    import os
    
    save_best_dir = os.path.join(args['root_dir'], "{}_seq{}_dis{}_best.pt".format(args['tag'], args['seq_len'], args['pred_len']))
    save_last_dir = os.path.join(args['root_dir'], "{}_seq{}_dis{}_last.pt".format(args['tag'], args['seq_len'], args['pred_len']))
    tensorboard_dir = os.path.join("./runs/", "tensorboard_{}_seq{}_dis{}".format(args['tag'], args['seq_len'], args['pred_len']))

    loss_fn = CustomLoss()
    
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
        test_for_check_per_epoch = test_loader
    )
    
    model.load_state_dict(torch.load(save_best_dir))

    # evaluation process
    test_loss, mse, rmse, mae, r2 = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
    )
    
    shot_num = ts_test.shot.iloc[0]
    df_shot = ts_test[ts_test.shot == shot_num].reset_index(drop = True)
    
    generate_shot_data_from_self(
        model,
        df_shot,
        seq_len,
        seq_len + pred_len,
        pred_len,
        cols_0D,
        cols_control,
        None,
        None,
        device,
        "shot number : {}".format(shot_num),
        save_dir = os.path.join("./result/", "{}_seq{}_dis{}_without_real_data.png".format(args['tag'], args['seq_len'], args['pred_len']))
    )
    
    generate_shot_data_from_real(
        model,
        df_shot,
        seq_len,
        seq_len + pred_len,
        pred_len,
        cols_0D,
        cols_control,
        None,
        None,
        device,
        "shot number : {}".format(shot_num),
        save_dir = os.path.join("./result/", "{}_seq{}_dis{}_with_real_data.png".format(args['tag'], args['seq_len'], args['pred_len']))
    )