import torch
import argparse
import numpy as np
import pandas as pd
from src.config import Config
from src.nn_env.utility import preparing_0D_dataset, get_range_of_output
from src.nn_env.dataset import DatasetFor0D
from src.nn_env.transformer_multi_step import Transformer
from src.nn_env.train_multi_step import train
from src.nn_env.loss import CustomLoss
from src.nn_env.evaluate_multi_step import evaluate
from src.nn_env.predict_multi_step import generate_shot_data_from_self
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="training NN based environment - Transformer : Multi-step")
parser.add_argument("--batch_size", type = int, default = 256)
parser.add_argument("--lr", type = float, default = 2e-4)
parser.add_argument("--gpu_num", type = int, default = 3)
parser.add_argument("--num_epoch", type = int, default = 128)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--verbose", type = int, default = 4)
parser.add_argument("--max_norm_grad", type = float, default = 1.0)
parser.add_argument("--root_dir", type = str, default = "./weights/")
parser.add_argument("--tag", type = str, default = "Transformer_multi_step")
parser.add_argument("--use_scaler", type = bool, default = True)
parser.add_argument("--scaler", type = str, default = 'Robust', choices = ['Standard', 'Robust', 'MinMax'])
parser.add_argument("--seq_len", type = int, default = 20)
parser.add_argument("--pred_len", type = int, default = 20)
parser.add_argument("--interval", type = int, default = 4)

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
    
    train_data = DatasetFor0D(ts_train.copy(deep = True), df_disruption, seq_len, seq_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl, True)
    valid_data = DatasetFor0D(ts_valid.copy(deep = True), df_disruption, seq_len, seq_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl, True)
    test_data = DatasetFor0D(ts_test.copy(deep = True), df_disruption, seq_len, seq_len, pred_len, cols_0D, cols_control, interval, scaler_0D, scaler_ctrl, True)
    
    print("train data : ", train_data.__len__())
    print("valid data : ", valid_data.__len__())
    print("test data : ", test_data.__len__())

    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = 4, shuffle = True, pin_memory = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = 4, shuffle = True, pin_memory = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 4, shuffle = True, pin_memory = True)
    
    # data range
    range_info = get_range_of_output(train_data.ts_data, cols_0D)
    
    # transformer model argument
    model = Transformer(
        n_layers = config.TRANSFORMER_CONF['n_layers'], 
        n_heads = config.TRANSFORMER_CONF['n_heads'], 
        dim_feedforward = config.TRANSFORMER_CONF['dim_feedforward'], 
        dropout = config.TRANSFORMER_CONF['dropout'],        
        RIN = config.TRANSFORMER_CONF['RIN'],
        input_0D_dim = len(cols_0D),
        input_ctrl_dim = len(cols_control),
        input_seq_len = seq_len,
        output_pred_len = pred_len,
        output_0D_dim = len(cols_0D),
        feature_dim = config.TRANSFORMER_CONF['feature_0D_dim'],
        range_info = range_info,
        noise_mean = config.TRANSFORMER_CONF['noise_mean'],
        noise_std = config.TRANSFORMER_CONF['noise_std']
    )

    model.summary()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=args['gamma'])
    
    import os
    
    save_best_dir = os.path.join(args['root_dir'], "{}_seq{}_dis{}_best.pt".format(args['tag'], args['seq_len'], args['pred_len']))
    save_last_dir = os.path.join(args['root_dir'], "{}_seq{}_dis{}_last.pt".format(args['tag'], args['seq_len'], args['pred_len']))
    tensorboard_dir = os.path.join("./runs/", "tensorboard_{}_seq{}_dis{}".format(args['tag'], args['seq_len'], args['pred_len']))

    # loss_fn = CustomLoss() 
    loss_fn = torch.nn.MSELoss(reduction = 'mean')
    
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
    
    # shot_num = ts_test.shot.iloc[0]
    # df_shot = ts_test[ts_test.shot == shot_num].reset_index(drop = True)
    
    shot_num = ts_train.shot.iloc[0]
    df_shot = ts_train[ts_train.shot == shot_num].reset_index(drop = True)
    
    generate_shot_data_from_self(
        model,
        df_shot,
        seq_len,
        seq_len,
        cols_0D,
        cols_control,
        scaler_0D,
        scaler_ctrl,
        device,
        "shot number : {}".format(shot_num),
        save_dir = os.path.join("./result/", "{}_seq{}_dis{}_without_real_data.png".format(args['tag'], args['seq_len'], args['pred_len']))
    )