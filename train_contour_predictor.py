import torch
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.config import Config
from src.GSsolver.model import ContourRegressor
from src.GSsolver.train_contour import train, evaluate
from src.GSsolver.dataset import PINNDataset

def parsing():
    parser = argparse.ArgumentParser(description="training contour-regressor based GS solver for plasma shape control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "contour")
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # objective : params control vs shape control
    parser.add_argument("--objective", type = str, default = "shape-control", choices = ['params-control', 'shape-control'])
    
    # training setup
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--num_epoch", type = int, default = 256)  
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--max_norm_grad", type = float, default = 1.0)
    parser.add_argument("--verbose", type = int, default = 4)
    
    # scheduler
    parser.add_argument("--step_size", type = int, default = 16)
    parser.add_argument("--gamma", type = float, default = 0.995)
    
    # geometrical properties
    parser.add_argument("--nx", type = int, default = 65)
    parser.add_argument("--ny", type = int, default = 65)
    
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
    
    config = Config()
    
    if args['objective'] == 'params-control':
        cols_0D = config.input_params['GS-solver-params-control']['state']
        cols_PFC = config.input_params['GS-solver-params-control']['control']

        args['tag'] = "{}_params-control".format(args['tag'])
    else:
        cols_0D = config.input_params['GS-solver']['state']
        cols_PFC = config.input_params['GS-solver']['control']
    
    df = pd.read_csv("./dataset/KSTAR_rl_GS_solver.csv")
    df_train, df_valid = train_test_split(df, test_size = 0.4, random_state=40)
    df_valid, df_test = train_test_split(df_valid, test_size=0.5, random_state=40)
    
    train_data = PINNDataset(df_train, cols_0D, cols_PFC)
    valid_data = PINNDataset(df_valid, cols_0D, cols_PFC)
    test_data = PINNDataset(df_test, cols_0D, cols_PFC)
    
    batch_size = args['batch_size']
    
    print("============= Dataset info =============")
    print("train data : {}".format(len(df_train)))
    print("valid data : {}".format(len(df_valid)))
    print("test data : {}".format(len(df_test)))
    
    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    
    params_dim = len(cols_0D)
    n_PFCs = len(cols_PFC)
    
    contour_regressor = ContourRegressor(args['nx'], args['ny'], params_dim, n_PFCs, 1.0, 2.5, -1.5, 1.5)
    contour_regressor.to(device)

    # optimizer
    contour_optimizer = torch.optim.AdamW(params = contour_regressor.parameters(), lr = args['lr'])
    
    # scheduler
    contour_scheduler = torch.optim.lr_scheduler.StepLR(contour_optimizer, step_size = args['step_size'], gamma = args['gamma'])
    contour_regressor.train()
    
    print("============= Training process =============")
    train(
        train_loader,
        valid_loader,
        contour_regressor,
        contour_optimizer,
        contour_scheduler,
        device,
        args['num_epoch'],
        verbose = args['verbose'],
        save_best_dir="./weights/{}_best.pt".format(args['tag']),
        save_last_dir="./weights/{}_last.pt".format(args['tag']),
        max_norm_grad=args['max_norm_grad'],
    )

    contour_regressor.eval()
    contour_regressor.load_state_dict(torch.load("./weights/{}_best.pt".format(args['tag'])))
    
    # evaluation
    print("=============== Evaluation ================")
    evaluate(
        test_loader,
        contour_regressor,
        device,
    )
    
    # test
    sample_loader = DataLoader(test_data, batch_size = 1, num_workers=1, pin_memory=False, drop_last=True, shuffle=True)
    
    data, target = next(iter(sample_loader))
    prediction = contour_regressor.compute_rzbdys(target.to(device), data['params'].to(device), data['PFCs'].to(device))
    rzbdys = data['rzbdys'].squeeze(0).numpy()
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize = (8,4))

    cen, rad = contour_regressor(target.to(device), data['params'].to(device), data['PFCs'].to(device))
    cen = cen.detach().squeeze(0).cpu().numpy()
    rad = rad.detach().squeeze(0).cpu().numpy()
    
    rc = 0.5 * (min(rzbdys[:,0]) + max(rzbdys[:,0]))
    ind = rzbdys[:,0].argmax()
    zc = rzbdys[ind, 1]
    rad_real = np.sqrt((rzbdys[:,0] - rc) ** 2 + (rzbdys[:,1] - zc) ** 2)
    theta = np.linspace(0,2*3.141,256)
    
    ax = axes[0]
    ax.plot(prediction[:,0], prediction[:,1], c = 'r')
    ax.plot(rzbdys[:,0], rzbdys[:,1], c = 'b')
    ax.scatter(rc, zc, marker = "o", c = 'k', label = "center-real")
    ax.scatter(cen[0], cen[1], marker = "o", c = 'r', label = "center-predict")
    ax.legend()
    
    ax = axes[1]
    ax.plot(theta, rad, c='r')
    ax.plot(theta, rad_real, c = 'b')
    
    fig.tight_layout()
    plt.savefig("./result/{}-test.png".format(args['tag']))
    