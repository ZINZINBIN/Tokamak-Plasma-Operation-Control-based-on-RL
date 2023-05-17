import torch
import torch.nn as nn
import numpy as np
from matplotlib.gridspec import GridSpec
from src.GSsolver.model import PINN
from src.GSsolver.util import draw_KSTAR_limiter

# torch device state
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:{}".format(1)
else:
    device = 'cpu'
    
# SSIM loss
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        # 입력 경계의 반사를 사용하여 상/하/좌/우에 입력 텐서를 추가로 채웁니다.
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # shape : (xh, xw) -> (xh + 2, xw + 2)
        x = self.refl(x) 
        # shape : (yh, yw) -> (yh + 2, yw + 2)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        
        # Loss function
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).sum()
    
if __name__ == "__main__":
    sample_data = np.load("./src/GSsolver/toy_dataset/g028911_004060.npz")
    
    psi = sample_data['psi']
    R = sample_data['R']
    Z = sample_data['Z']
    
    print("psi : ", psi.shape)
    print("R : ", R.shape)
    print("Z : ", Z.shape)

    ip = (-1) * 800828
    q95 = 3.330114
    kappa = 1.75662
    betap = 1.009343
    betan = 1.902915
    tribot = 0.8311241
    tritop = 0.359658
    li = 0.841751
    
    PCPF1U = 4282.0
    PFPF1L = 0
    PCPF2U = 4543.6
    PFPF2L = 0
    PCPF3U = (-1) * 5441.8
    PFPF3L = (-1) * 5539.4
    PCPF4U = (-1) * 9353.0
    PFPF4L = (-1) * 10078.6
    PCPF5U = (-1) * 3643.2
    PFPF5L = (-1) * 4900.2
    PCPF6U = 4374.0
    PFPF6L = 5211.4
    PCPF7U = 2316.8
    PFPF7L = 0
    
    x_param = torch.Tensor([ip, betap, q95, li])
    x_PFCs = torch.Tensor([
        PCPF1U,
        PFPF1L,
        PCPF2U,
        PFPF2L,
        PCPF3U,
        PFPF3L,
        PCPF4U,
        PFPF4L,
        PCPF5U,
        PFPF5L,
        PCPF6U,
        PFPF6L,
        PCPF7U,
        PFPF7L
    ])
    
    # setup
    alpha_m = 2.0
    alpha_n = 1.0
    beta_m = 2.0
    beta_n = 1.0
    lamda = 1.0
    beta = 0.5
    Rc = 1.8
    
    params_dim = 4
    n_PFCs = 14
    hidden_dim = 64
    
    # model load
    model = PINN(R,Z,Rc, params_dim, n_PFCs, hidden_dim, alpha_m, alpha_n, beta_m, beta_n, lamda, beta, 65, 65, False, 1e6)
    model.to(device)

    # input data
    x_param = x_param.unsqueeze(0)
    x_PFCs = x_PFCs.unsqueeze(0)
    
    # target data
    target = torch.from_numpy(psi).unsqueeze(0).float()
    
    # loss function
    loss_mse = torch.nn.MSELoss(reduce='sum')
    loss_ssim = SSIM()
    
    # optimizer
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = 1e-2)
    
    model.train()
    
    best_loss = np.inf
    
    for epoch in range(1000):
        optimizer.zero_grad()
        psi_p = model(x_param.to(device), x_PFCs.to(device))
        
        mse = loss_mse(psi_p, target.to(device))
        gs_loss = model.compute_GS_loss(psi_p)
        constraint_loss = model.compute_constraint_loss(psi_p.detach(), ip)
        
        loss = gs_loss * 0.1 + mse + constraint_loss * 0.1
        
        loss.backward()
        optimizer.step()
        
        ssim = loss_ssim(psi_p.detach(), target.to(device))
        # constraint_loss = model.compute_constraint_loss(psi_p.detach(), ip)
        
        if epoch % 64 ==0:
            print("# epoch: {} | mse: {:.3f} | constraint: {:.3f} | gs loss: {:.3f} | SSIM : {:.3f}".format(
                epoch, mse.cpu().item(), constraint_loss.cpu().item(), gs_loss.cpu().item(), ssim.cpu().item()
            ))

        if mse.detach().cpu().item() < best_loss:
           torch.save(model.state_dict(), "./PINN_best.pt")
           best_loss = mse.detach().cpu().item()
    
    model.eval()
    model.load_state_dict(torch.load("./PINN_best.pt"))
    
    psi_p = model(x_param.to(device), x_PFCs.to(device))
    psi_p_np = psi_p.detach().cpu().squeeze(0).numpy()
    
    gs_loss = model.compute_GS_loss(psi_p)
    print("gs loss for terminal state : {:.3f}".format(gs_loss.detach().cpu().item()))
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
    Jphi = model.compute_Jphi(psi_p)
    Jphi = Jphi.detach().cpu().squeeze(0).numpy()
    
    x_psi, ffprime = model.compute_ffprime()
    x_psi, pprime = model.compute_pprime()
    
    x_psi, pressure = model.compute_pressure_psi()    
    x_psi, Jphi1D = model.compute_Jphi_psi(psi_p)
    x_psi, q = model.compute_q_psi(psi_p)
    
    (r_axis, z_axis), _ = model.find_axis(psi_p, eps = 1e-4)
    
    fig = plt.figure(figsize = (16, 5))
    fig.suptitle("PINN test result : $\Psi$, $J_\phi$, $P(\psi)$ profile")
    gs = GridSpec(nrows = 2, ncols = 4)
    
    ax = fig.add_subplot(gs[:,0])
    ax.contourf(R,Z,psi_p_np)
    ax.plot(r_axis, z_axis, "x", label = 'axis')
    ax = draw_KSTAR_limiter(ax)
    norm = colors.Normalize(vmin = psi_p_np.min(), vmax = psi_p_np.max())
    map = cm.ScalarMappable(norm=norm)
    fig.colorbar(map)
    ax.set_xlabel("R[m]")
    ax.set_ylabel("Z[m]")
    ax.set_title('Poloidal flux ($\psi$)')
    
    ax = fig.add_subplot(gs[:,1])
    ax.contourf(R,Z,Jphi)
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
    ax[0].contourf(R,Z,psi)
    ax[0] = draw_KSTAR_limiter(ax[0])
    ax[1].contourf(R,Z,psi_p_np)
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
    
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.contourf(R,Z,psi_p_norm.detach().squeeze(0).cpu().numpy())
    ax = draw_KSTAR_limiter(ax)
    ax.set_xlabel("R[m]")
    ax.set_ylabel("Z[m]")
    ax.set_title('psi-norm')
    
    fig.tight_layout()
    norm = colors.Normalize(vmin = psi_p_norm.min(), vmax = psi_p_norm.max())
    map = cm.ScalarMappable(norm=norm)
    fig.colorbar(map, ax = ax)
    
    plt.savefig("./PINN_psi_norm.png")
    
    