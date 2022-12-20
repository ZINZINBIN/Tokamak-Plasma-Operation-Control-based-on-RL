import torch
import argparse
import numpy as np
import pandas as pd
from src.nn_env.dataset import DatasetFor0D
from src.nn_env.model import CnnLSTM, TStransformer
from src.nn_env.train import train
from src.nn_env.loss import CustomLoss
from src.nn_env.evaluate import evaluate
from src.nn_env.predict import real_time_predict, generate_shot_data
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="training NN based environment")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--num_epoch", type = int, default = 32)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--verbose", type = int, default = 4)
parser.add_argument("--max_norm_grad", type = float, default = 1.0)
parser.add_argument("--root_dir", type = str, default = "./weights/")
parser.add_argument("--tag", type = str, default = "TStransformer")
parser.add_argument("--seq_len", type = int, default = 21)
parser.add_argument("--pred_len", type = int, default = 3)
parser.add_argument("--interval", type = int, default = 3)
parser.add_argument("--dist", type = int, default = 0)

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
        '\\q0', '\\q95', '\\ipmhd', '\\kappa', 
        '\\tritop', '\\tribot','\\betap','\\betan',
        '\\li', '\\WTOT_DLM03'
    ]
    
    # else diagnostics
    cols_diag = [
        '\\ne_inter01', '\\ne_tci01', '\\ne_tci02', '\\ne_tci03', '\\ne_tci04', '\\ne_tci05',
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
    
    ts_cols = cols_0D + cols_control

    # float type
    for col in ts_cols:
        df[col] = df[col].astype(np.float32)

    # train / valid / test data split
    from sklearn.model_selection import train_test_split
    shot_list = np.unique(df.shot.values)

    shot_train, shot_test = train_test_split(shot_list, test_size = 0.2, random_state = 42)
    shot_train, shot_valid = train_test_split(shot_train, test_size = 0.2, random_state = 42)

    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for shot in shot_train:
        df_train = pd.concat([df_train, df[df.shot == shot]], axis = 0)

    for shot in shot_valid:
        df_valid = pd.concat([df_valid, df[df.shot == shot]], axis = 0)

    for shot in shot_test:
        df_test = pd.concat([df_test, df[df.shot == shot]], axis = 0)

    scaler = RobustScaler()
    df_train[ts_cols] = scaler.fit_transform(df_train[ts_cols].values)
    df_valid[ts_cols] = scaler.transform(df_valid[ts_cols].values)
    df_test[ts_cols] = scaler.transform(df_test[ts_cols].values)

    ts_train = df_train
    ts_valid = df_valid
    ts_test = df_test
    
    seq_len = args['seq_len']
    pred_len = args['pred_len']
    interval = args['interval']
    dist = args['dist']
    batch_size = args['batch_size']
    
    pred_cols = cols_0D
    
    train_data = DatasetFor0D(ts_train, df_disruption, seq_len, pred_len, dist, cols_0D, cols_control, pred_cols, interval, scaler = None)
    valid_data = DatasetFor0D(ts_valid, df_disruption, seq_len, pred_len, dist, cols_0D, cols_control, pred_cols, interval, scaler = None)
    test_data = DatasetFor0D(ts_test, df_disruption, seq_len, pred_len, dist, cols_0D, cols_control, pred_cols, interval, scaler = None)
    
    print("train data : ", train_data.__len__())
    print("valid data : ", valid_data.__len__())
    print("test data : ", test_data.__len__())

    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = 8, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = 8, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 8, shuffle = True)

    if args['tag'] == 'TStransformer':
        
        model = TStransformer(
            n_features = len(cols_0D + cols_control), 
            feature_dims = 128, 
            max_len = seq_len, 
            n_layers = 4, 
            n_heads = 8, 
            dim_feedforward = 512, 
            dropout = 0.25, 
            mlp_dim = 64, 
            pred_len = pred_len,
            output_dim = len(pred_cols)
        )
        
    else:
        model = CnnLSTM(
            seq_len = seq_len,
            pred_len = pred_len,
            col_dim = len(cols_0D + cols_control),
            conv_dim=64,
            conv_kernel = 3,
            conv_stride=1,
            conv_padding = 1,
            output_dim = len(pred_cols)
        )

    model.summary()
    
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=args['gamma'])
    
    import os
    save_best_dir = os.path.join(args['root_dir'], args['tag'] + "_best.pt")
    save_last_dir = os.path.join(args['root_dir'], args['tag'] + "_last.pt")

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
    )
    
    
    model.load_state_dict(torch.load(save_best_dir))

    # evaluation process
    test_loss = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
    )
    
    shot_num = test_data.ts_data.shot.iloc[0]
    df_shot = test_data.ts_data[test_data.ts_data.shot == shot_num].reset_index(drop = True)
    
    # real-time prediction
    real_time_predict(
        model,
        df_shot,
        seq_len,
        pred_len,
        dist,
        cols_0D + cols_control,
        pred_cols,
        None,
        device,
        "shot number : {}".format(shot_num),
        save_dir = "./result/nn_env_feedforward.png"
    )
    
    generate_shot_data(
        model,
        df_shot,
        seq_len,
        pred_len,
        dist,
        cols_0D,
        cols_control,
        device,
        "shot number : {}".format(shot_num),
        save_dir = "./result/nn_env_without_feedforward.png"
    )