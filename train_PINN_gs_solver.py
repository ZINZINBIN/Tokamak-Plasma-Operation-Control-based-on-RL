import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
from src.GSsolver.model import PINN
from src.GSsolver.GradShafranov import compute_plasma_region
from src.GSsolver.util import draw_KSTAR_limiter, modify_resolution
from src.GSsolver.train import train
from src.GSsolver.loss import SSIM
from src.GSsolver.dataset import PINNDataset

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:{}".format(0)
else:
    device = 'cpu'
    
cols_PFC = ['\PCPF1U', '\PCPF2U', '\PCPF3U', '\PCPF3L', '\PCPF4U','\PCPF4L', '\PCPF5U', '\PCPF5L', '\PCPF6U', '\PCPF6L', '\PCPF7U']
cols_0D = ['\\ipmhd', '\\q95','\\betap', '\li',]
    
if __name__ == "__main__":
    
    df = pd.read_csv("./dataset/KSTAR_rl_GS_solver.csv")
    df_train, df_valid = train_test_split(df, test_size = 0.3)
    train_data = PINNDataset(df_train, cols_0D, cols_PFC)
    valid_data = PINNDataset(df_valid, cols_0D, cols_PFC)
    
    train_loader = DataLoader(train_data, batch_size = 32, num_workers=4, pin_memory=True, drop_last=True, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = 32, num_workers=4, pin_memory=True, drop_last=True, shuffle = True)
    
    sample_data = np.load("./src/GSsolver/toy_dataset/g028911_004060.npz")
    R = sample_data['R']
    Z = sample_data['Z']
    psi = sample_data['psi']
    
    ip = (-1) * 800828
    q95 = 3.330114
    kappa = 1.75662
    betap = 1.009343
    betan = 1.902915
    tribot = 0.8311241
    tritop = 0.359658
    li = 0.841751
    
    PCPF1U = 4282.0
    PCPF2U = 4543.6
    PCPF3U = (-1) * 5441.8
    PFPF3L = (-1) * 5539.4
    PCPF4U = (-1) * 9353.0
    PFPF4L = (-1) * 10078.6
    PCPF5U = (-1) * 3643.2
    PFPF5L = (-1) * 4900.2
    PCPF6U = 4374.0
    PFPF6L = 5211.4
    PCPF7U = 2316.8
    
    x_param = torch.Tensor([ip, betap, q95, li])
    x_PFCs = torch.Tensor([
        PCPF1U,
        PCPF2U,
        PCPF3U,
        PFPF3L,
        PCPF4U,
        PFPF4L,
        PCPF5U,
        PFPF5L,
        PCPF6U,
        PFPF6L,
        PCPF7U
    ])
    
    # input data
    x_param = x_param.unsqueeze(0)
    x_PFCs = x_PFCs.unsqueeze(0)
    
    # target data
    target = torch.from_numpy(psi).unsqueeze(0).float()
    
    # setup
    alpha_m = 2.0
    alpha_n = 2.0
    beta_m = 2.0
    beta_n = 1.0
    lamda = 1e-1
    beta = 0.5
    Rc = 1.8
    
    params_dim = 4
    n_PFCs = 11
    hidden_dim = 64
    
    # model load
    model = PINN(R,Z,Rc, params_dim, n_PFCs, hidden_dim, alpha_m, alpha_n, beta_m, beta_n, lamda, beta, 65, 65, False, 1e6)
    model.to(device)

    # loss function
    loss_mse = torch.nn.MSELoss(reduction='mean')
    loss_mask = torch.nn.MSELoss(reduction = 'mean')
    loss_ssim = SSIM()
    
    # optimizer
    optimizer = torch.optim.RMSprop(params = model.parameters(), lr = 1e-2)
    
    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 8, gamma = 0.995)
    
    # weights for loss
    weights = {
        "GS_loss" : 0.1,
        "Constraint_loss" : 0.1 
    }
    
    model.train()
    
    train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        device,
        1024,
        verbose = 32,
        save_best_dir="./PINN_best.pt",
        save_last_dir="./PINN_last.pt",
        max_norm_grad=1.0,
        weights=weights
    )
    
    model.eval()
    model.load_state_dict(torch.load("./PINN_best.pt"))
    
    psi_p = model(x_param.to(device), x_PFCs.to(device))
    psi_p_np = psi_p.detach().cpu().squeeze(0).numpy()
    
    gs_loss = model.compute_GS_loss(psi_p)
    
    print("mse loss : {:.3f}".format(loss_mse(psi_p, target.to(device)).detach().cpu().item()))
    print("gs loss : {:.3f}".format(gs_loss.detach().cpu().item()))
    print("alpha_m : ", model.alpha_m.detach().cpu().item())
    print("alpha_n : ", model.alpha_n.detach().cpu().item())
    print("beta_m : ", model.beta_m.detach().cpu().item())
    print("beta_n : ", model.beta_n.detach().cpu().item())
    print("beta : ", model.beta.detach().cpu().item())
    print("lamda : ", model.lamda.detach().cpu().item())
    print("Ip scale : ", model.Ip_scale.detach().cpu().item())
    
    import matplotlib.pyplot as plt
    from matplotlib import colors, cm
    
    # toroidal current profile
    Jphi = model.compute_Jphi_GS(psi_p)
    Jphi *= model.Ip_scale
    Jphi = Jphi.detach().cpu().squeeze(0).numpy()
    
    x_psi, ffprime = model.compute_ffprime()
    x_psi, pprime = model.compute_pprime()
    
    x_psi, pressure = model.compute_pressure_psi()    
    x_psi, Jphi1D = model.compute_Jphi_psi(psi_p)
    x_psi, q = model.compute_q_psi(psi_p)
    
    (r_axis, z_axis), _ = model.find_axis(psi_p, eps = 1e-4)
    
    psi_a, psi_b = model.norms.predict_critical_value(psi_p)
        
    print("psi axis : ", psi_a.detach().cpu().item())
    print("psi bndy : ", psi_b.detach().cpu().item())
    
    fig = plt.figure(figsize = (16, 5))
    fig.suptitle("PINN test result : $\Psi$, $J_\phi$, $P(\psi)$ profile")
    gs = GridSpec(nrows = 2, ncols = 4)
    
    ax = fig.add_subplot(gs[:,0])
    ax.contourf(R,Z,psi_p_np, levels = 32)
    ax.plot(r_axis, z_axis, "x", c = 'r',label = 'axis')
    ax = draw_KSTAR_limiter(ax)
    norm = colors.Normalize(vmin = psi_p_np.min(), vmax = psi_p_np.max())
    map = cm.ScalarMappable(norm=norm)
    fig.colorbar(map)
    ax.set_xlabel("R[m]")
    ax.set_ylabel("Z[m]")
    ax.set_title('Poloidal flux ($\psi$)')
    
    ax = fig.add_subplot(gs[:,1])
    ax.contourf(R,Z,Jphi, levels = 32)
    ax = draw_KSTAR_limiter(ax)
    norm = colors.Normalize(vmin = Jphi.min(), vmax = Jphi.max())
    map = cm.ScalarMappable(norm=norm)
    fig.colorbar(map)
    ax.set_xlabel("R[m]")
    ax.set_title('Toroidal current ($J_\phi$)')
    
    ax = fig.add_subplot(gs[0,2])
    ax.plot(x_psi, ffprime, 'r-', label = "$FF'(\psi)$")
    ax.plot(x_psi, pprime, 'b-', label = "$P(\psi)'$")    
    ax.set_xlabel("Normalized $\psi$")
    ax.set_ylabel("Relative value")
    ax.legend()
    
    ax = fig.add_subplot(gs[1,2])
    ax.plot(x_psi, Jphi1D, 'r-', label = '$J_\phi$')
    ax.set_xlabel('Normalized $\psi$')
    ax.set_ylabel("Relative value")
    ax.legend()
    
    ax = fig.add_subplot(gs[0,3])
    ax.plot(x_psi, q, 'r-', label = '$q(\psi)$')
    ax.set_xlabel("Normalized $\psi$")
    ax.set_ylabel("q($\psi$)")
    ax.legend()    
    
    ax = fig.add_subplot(gs[1,3])
    ax.plot(x_psi, pressure, 'r-', label = '$P(\psi)$')
    ax.set_xlabel("Normalized $\psi$")
    ax.set_ylabel("Relative value")
    ax.legend()

    fig.tight_layout()
    plt.savefig("./PINN_profile.png")
    
    fig, ax = plt.subplots(2,1,figsize=(4,8))
    ax[0].contourf(R,Z,psi, levels = 32)
    ax[0] = draw_KSTAR_limiter(ax[0])
    
    ax[1].contourf(R,Z,psi_p_np, levels = 32)
    ax[1] = draw_KSTAR_limiter(ax[1])
    
    ax[0].set_xlabel("R[m]")
    ax[0].set_ylabel("Z[m]")
    ax[0].set_title("EFIT-psi")
    
    ax[1].set_xlabel("R[m]")
    ax[1].set_ylabel("Z[m]")
    ax[1].set_title('PINN-psi')
    
    fig.tight_layout()
    
    norm = colors.Normalize(vmin = psi.min(), vmax = psi.max())
    map = cm.ScalarMappable(norm=norm)
    fig.colorbar(map, ax = ax)
    plt.savefig("./PINN_psi.png")
    
    psi_p_norm = model.norms(psi_p)
    
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].contourf(R,Z,psi_p_norm.detach().squeeze(0).cpu().numpy(), levels = 32)
    ax[0] = draw_KSTAR_limiter(ax[0])
    ax[0].set_xlabel("R[m]")
    ax[0].set_ylabel("Z[m]")
    ax[0].set_title('psi-norm')
    
    norm = colors.Normalize(vmin = psi_p_norm.min(), vmax = psi_p_norm.max())
    map = cm.ScalarMappable(norm=norm)
    fig.colorbar(map, ax = ax[0])
    
    mask = model.compute_plasma_region(psi_p).detach().cpu().squeeze(0).numpy()
    ax[1].contourf(R,Z,mask)
    ax[1] = draw_KSTAR_limiter(ax[1])
    ax[1].set_xlabel("R[m]")
    ax[1].set_ylabel("Z[m]")
    ax[1].set_title('mask')
    
    norm = colors.Normalize(vmin = mask.min(), vmax = mask.max())
    map = cm.ScalarMappable(norm=norm)
    fig.colorbar(map, ax = ax[1])
    
    fig.tight_layout()
    plt.savefig("./PINN_psi_norm.png")
    