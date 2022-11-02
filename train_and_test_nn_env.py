import torch
import argparse
import numpy as np
import pandas as pd
from src.CustomDataset import DatasetFor0D
from src.nn_env import ConvLSTM
from src.train_env import train
from src.loss import CustomLoss
from src.evaluate_env import evaluate
from src.predict_env import real_time_predict
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="training NN based environment")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--gpu_num", type = int, default = 0)
parser.add_argument("--num_epoch", type = int, default = 16)
parser.add_argument("--gamma", type = float, default = 0.95)
parser.add_argument("--verbose", type = int, default = 4)
parser.add_argument("--max_norm_grad", type = float, default = 1.0)
parser.add_argument("--root_dir", type = str, default = "./weights/")
parser.add_argument("--tag", type = str, default = "CNN_LSTM")
parser.add_argument("--seq_len", type = int, default = 21)
parser.add_argument("--pred_len", type = int, default = 7)
parser.add_argument("--interval", type = int, default = 7)
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

    # nan interpolation
    df.interpolate(method = 'linear', limit_direction = 'forward')

    # columns for use
    ts_cols = [
        '\\q95', '\\ipmhd', '\\kappa', 
        '\\tritop', '\\tribot','\\betap','\\betan',
        '\\li', '\\WTOT_DLM03'
    ]

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
    
    cols = ts_cols
    pred_cols = ts_cols
    
    batch_size = args['batch_size']
    
    train_data = DatasetFor0D(ts_train, seq_len, pred_len, dist, cols, pred_cols, interval, scaler = None)
    valid_data = DatasetFor0D(ts_valid, seq_len, pred_len, dist, cols, pred_cols, interval, scaler = None)
    test_data = DatasetFor0D(ts_test, seq_len, pred_len, dist, cols, pred_cols, interval, scaler = None)

    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers =8, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = 8, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 8, shuffle = True)

    model = ConvLSTM(
        seq_len = seq_len,
        pred_len = pred_len,
        col_dim = len(cols),
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
    
    shot_num = ts_test.shot.iloc[0]
    df_shot = ts_test[ts_test.shot == shot_num].reset_index(drop = True)
    
    # real-time prediction
    real_time_predict(
        model,
        df_shot,
        seq_len,
        pred_len,
        dist,
        cols,
        pred_cols,
        interval,
        None,
        device,
        "shot number : {}".format(shot_num),
        save_dir = "./result/nn_env_performance.png"
    )