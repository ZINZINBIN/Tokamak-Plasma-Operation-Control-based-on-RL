import torch
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.config import Config
from src.GSsolver.model import PINN, ContourRegressor
from src.GSsolver.util import plot_PINN_profile, plot_PINN_comparison
from src.GSsolver.train import train
from src.GSsolver.loss import SSIM
from src.GSsolver.dataset import PINNDataset
from src.GSsolver.evaluate import evaluate

def parsing():
    parser = argparse.ArgumentParser(description="training PINN based GS solver for plasma shape control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "PINN")
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # use contour regressor
    parser.add_argument("--use_contour_regressor", type = bool, default = True)
    parser.add_argument("--contour_regressor_weight", type = str, default = "./weights/contour_best.pt")
    
    # objective : params control vs shape control
    parser.add_argument("--objective", type = str, default = "shape-control", choices = ['params-control', 'shape-control'])
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # training setup
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--num_epoch", type = int, default = 1024)  
    parser.add_argument("--lr", type = float, default = 1e-2)
    parser.add_argument("--max_norm_grad", type = float, default = 1.0)
    parser.add_argument("--verbose", type = int, default = 32)
    
    # scheduler
    parser.add_argument("--step_size", type = int, default = 16)
    parser.add_argument("--gamma", type = float, default = 0.995)
    
    # pprime and ffprime profile
    parser.add_argument("--alpha_m", type = int, default = 2)
    parser.add_argument("--alpha_n", type = int, default = 1)
    parser.add_argument("--beta_m", type = int, default = 2)
    parser.add_argument("--beta_n", type = int, default = 1)
    parser.add_argument("--beta", type = float, default = 0.5)
    parser.add_argument("--lamda", type = float, default = 1.0)
    
    # geometrical properties
    parser.add_argument("--Rc", type = float, default = 1.8)
    parser.add_argument("--nx", type = int, default = 65)
    parser.add_argument("--ny", type = int, default = 65)
    
    # model setup
    parser.add_argument("--hidden_dim", type = int, default = 128)
 
    # loss weight
    parser.add_argument("--GS_loss", type = float, default = 1.0)
    parser.add_argument("--Constraint_loss", type = float, default = 1.0)
  
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
    
    # samples for checking profile
    sample_loader = DataLoader(test_data, batch_size = 1, num_workers=1, pin_memory=False, drop_last=True, shuffle=True)
    
    data, target = next(iter(sample_loader))
    
    Ip = data['Ip']
    x_param = data['params']
    x_PFCs = data['PFCs']
    
    sample_data = np.load("./src/GSsolver/toy_dataset/g028911_004060.npz")
    R = sample_data['R']
    Z = sample_data['Z']
    
    # target data
    psi = target.squeeze(0).numpy()
    
    # setup
    alpha_m = args['alpha_m']
    alpha_n = args['alpha_n']
    beta_m = args['beta_m']
    beta_n = args['beta_n']
    lamda = args['lamda']
    beta = args['beta']
    Rc = args['Rc']
    
    params_dim = len(cols_0D)
    n_PFCs = len(cols_PFC)
    hidden_dim = args['hidden_dim']
    
    # model load
    model = PINN(R,Z,Rc, params_dim, n_PFCs, hidden_dim, alpha_m, alpha_n, beta_m, beta_n, lamda, beta, args['nx'], args['ny'])
    model.to(device)
    
    # addition : contour regression model
    contour_regressor = ContourRegressor(65, 65, params_dim, n_PFCs, 1.0, 4.0, -2.0, 2.0)
    contour_regressor.to(device)

    # loss function
    loss_mse = torch.nn.MSELoss(reduction='mean')
    loss_mask = torch.nn.MSELoss(reduction = 'mean')
    loss_ssim = SSIM()
    
    # optimizer
    optimizer = torch.optim.RMSprop(params = model.parameters(), lr = args['lr'])
    contour_optimizer = torch.optim.AdamW(params = contour_regressor.parameters(), lr = args['lr'])
    
    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['step_size'], gamma = args['gamma'])
    contour_scheduler = torch.optim.lr_scheduler.StepLR(contour_optimizer, step_size = args['step_size'], gamma = args['gamma'])
    
    # weights for loss
    weights = {
        "GS_loss" : args['GS_loss'],
        "Constraint_loss" : args['Constraint_loss']
    }
    
    model.train()
    
    print("============= Training process =============")
    train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        device,
        args['num_epoch'],
        verbose = args['verbose'],
        save_best_dir="./weights/{}_best.pt".format(args['tag']),
        save_last_dir="./weights/{}_last.pt".format(args['tag']),
        max_norm_grad=args['max_norm_grad'],
        weights=weights,
        test_for_check = None,
        contour_regressor=None, # contour_regressor,
        contour_optimizer=None, # contour_optimizer,
        contour_scheduler=None, # contour_scheduler,
        contour_save_best_dir=None,
        contour_save_last_dir=None
    )
    
    model.eval()
    model.load_state_dict(torch.load("./weights/{}_best.pt".format(args['tag'])))
    
    if args['use_contour_regressor']:
        contour_regressor.eval()
        contour_regressor.load_state_dict(torch.load(args['contour_regressor_weight']))
    
    # evaluation
    print("=============== Evaluation ================")
    evaluate(
        test_loader,
        model,
        device,
        weights,
        None
    )
    
    # visualization process
    # PINN profile
    print("============= Visualization ==============")
    plot_PINN_profile(model, data, device, "./result", tag = args['tag'], contour_regressor=contour_regressor)
    
    # Comparsion between real psi and PINN psi
    plot_PINN_comparison(model, psi, data, device, "./result", tag = args['tag'], contour_regressor=contour_regressor)