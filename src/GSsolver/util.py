import torch, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp2d
from matplotlib import colors, cm
from matplotlib.pyplot import Axes
from src.GSsolver.KSTAR_setup import limiter_shape
from src.GSsolver.model import AbstractPINN, ContourRegressor
from matplotlib.gridspec import GridSpec
from typing import Dict, Union, Optional

# draw KSTAR limiter (External boundary)
def draw_KSTAR_limiter(ax:Axes):
    ax.plot(limiter_shape[:,0], limiter_shape[:,1], 'black')
    ax.set_xlim([1.0, 2.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel('R[m]')
    ax.set_ylabel('Z[m]')
    return ax

def modify_resolution(psi : np.ndarray, R : np.ndarray, Z : np.ndarray, n_grid : int = 128):
    if n_grid % 2 == 0:
        n_grid += 1
    
    if min(R.shape[0], R.shape[1]) > n_grid:
        return ValueError("argument n_grid should be greater than the current grid number")
    
    interp_fn = interp2d(R,Z,psi)
    
    R_new = np.linspace(R.min(), R.max(), n_grid, endpoint = True)
    Z_new = np.linspace(Z.min(), Z.max(), n_grid, endpoint = True)
    
    RR, ZZ = np.meshgrid(R_new, Z_new)
    PSI = interp_fn(R_new,Z_new).reshape(n_grid, n_grid)
    return RR,ZZ,PSI

def compute_shape_parameters(rzbdy : Union[torch.Tensor, np.ndarray]):
    
    big_ind = 1
    small_ind = 1

    len2 = len(rzbdy)

    for i in range(len2-1):

        if (rzbdy[i,1] > rzbdy[big_ind,1]):
            big_ind = i

        if (rzbdy[i,1] < rzbdy[small_ind,1]):
            small_ind = i

    a = (max(rzbdy[:,0]) - min(rzbdy[:,0])) * 0.5
    R = (max(rzbdy[:,0]) + min(rzbdy[:,0])) * 0.5

    r_up = rzbdy[big_ind,0]
    r_low = rzbdy[small_ind,0]
    
    k = (rzbdy[big_ind,1]-rzbdy[small_ind,1])/a * 0.5
    triu = (R-r_up)/a
    tril = (R-r_low)/a
    
    return k, triu, tril, R, a

# plot predicted flux, 2D current profile and 1D profile of pressure, current and safety-factor
def plot_PINN_profile(model : AbstractPINN, data : Dict, device : str = "cpu", save_dir : str = "./result/", tag : str = "PINN", contour_regressor : Optional[ContourRegressor] = None):
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    x_param = data['params']
    x_PFCs = data['PFCs']
    Ip = data['Ip']
    betap = data['betap']
    
    model.eval()
    psi_p = model(x_param.to(device), x_PFCs.to(device))
    psi_p_np = psi_p.detach().cpu().squeeze(0).numpy()
    
    # compute GS loss : check the physical consistency
    gs_loss = model.compute_GS_loss(psi_p)
    Ip_loss = model.compute_constraint_loss_Ip(psi_p, Ip.to(device))
    betap_loss = model.compute_constraint_loss_betap(psi_p, Ip.to(device), betap.to(device))
    
    # compute beta and Ip : check the physical consistency for constraint
    Ip_compute = model.compute_plasma_current(psi_p).detach().cpu().squeeze(0).item()
    betap_compute = model.compute_betap(psi_p, Ip.to(device)).detach().cpu().squeeze(0).item()
    
    # positional information
    R = model.R2D
    Z = model.Z2D
    
    # toroidal current profile
    Jphi = model.compute_Jphi(psi_p)
    Jphi = Jphi.detach().cpu().squeeze(0).numpy()
    
    # presure profile and FF profile
    x_psi, ffprime = model.compute_ffprime()
    x_psi, pprime = model.compute_pprime()
    
    x_psi, pressure = model.compute_pressure_psi()    
    x_psi, Jphi1D = model.compute_Jphi_psi(psi_p)
    x_psi, q = model.compute_q_psi(psi_p)
    
    # X-point and O-point
    (r_axis, z_axis), _ = model.find_axis(psi_p, eps = 1e-4)
    psi_a, psi_b = model.norms.predict_critical_value(psi_p)
    
    # console : model information
    print("============= model information =============")
    print("PINN loss(GS loss) : {:.3f}".format(gs_loss.detach().cpu().item()))
    print("Constraint loss(Ip) : {:.3f}".format(Ip_loss.detach().cpu().item()))
    print("Constraint loss(betap) : {:.3f}".format(betap_loss.detach().cpu().item()))
    print("Ip : {:.3f}(MA) | Ip computed : {:.3f}(MA)".format(Ip.abs().item(), Ip_compute))
    print("betap : {:.3f} | betap computed : {:.3f}".format(betap.item(), betap_compute))
    print("alpha_m : {:.3f}".format(model.alpha_m))
    print("alpha_n : {:.3f}".format(model.alpha_n))
    print("beta_m : {:.3f}".format(model.beta_m))
    print("beta_n : {:.3f}".format(model.beta_n))
    print("beta : {:.3f}".format(model.beta.detach().cpu().item()))
    print("lamda : {:.3f}".format(model.lamda.detach().cpu().item()))
    print("grid number : ({},{})".format(model.nx, model.ny))
    print("psi axis : {:.3f}".format(psi_a.detach().cpu().item()))
    print("psi bndry : {:.3f}".format(psi_b.detach().cpu().item()))
    
    # plot the profile
    fig = plt.figure(figsize = (16, 5))
    fig.suptitle("PINN test result : $\Psi$, $J_\phi$, $P(\psi)$ profile")
    gs = GridSpec(nrows = 2, ncols = 4)
    
    ax = fig.add_subplot(gs[:,0])
    ax.contourf(R,Z,psi_p_np, levels = 32)
    ax.plot(r_axis, z_axis, "x", c = "r", label = 'magnetic axis')
    ax = draw_KSTAR_limiter(ax)
    norm = colors.Normalize(vmin = psi_p_np.min(), vmax = psi_p_np.max())
    map = cm.ScalarMappable(norm=norm)
    fig.colorbar(map)
    ax.set_xlabel("R[m]")
    ax.set_ylabel("Z[m]")
    ax.set_title('Poloidal flux ($\psi$)')
    
    if contour_regressor is not None:
        contour_regressor.eval()
        bndy = contour_regressor.compute_rzbdys(psi_p, x_param.to(device), x_PFCs.to(device))
        ax.plot(bndy[:,0], bndy[:,1], c = 'r', linewidth = 1.5)
    
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
    plt.savefig(os.path.join(save_dir, "{}_profile.png".format(tag)))
    
    # normalized psi and mask for free boundary solver
    psi_p_norm = model.norms(psi_p)
    
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].contourf(R,Z,psi_p_norm.detach().squeeze(0).cpu().numpy(), levels = 32)
    ax[0] = draw_KSTAR_limiter(ax[0])
    ax[0].set_xlabel("R[m]")
    ax[0].set_ylabel("Z[m]")
    ax[0].set_title('psi-norm')
    
    if contour_regressor is not None:
        contour_regressor.eval()
        bndy = contour_regressor.compute_rzbdys(psi_p, x_param.to(device), x_PFCs.to(device))
        ax[0].plot(bndy[:,0], bndy[:,1], c = 'r', linewidth = 1.5)
    
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
    plt.savefig(os.path.join(save_dir, "{}_psi_norm.png".format(tag)))
    
    
# plot the PINN result and real magnetic flux
def plot_PINN_comparison(model : AbstractPINN, psi:Union[torch.Tensor, np.ndarray], data : Dict, device : str = "cpu", save_dir : str = "./result/", tag : str = "PINN", contour_regressor : Optional[ContourRegressor] = None):
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    x_param = data['params']
    x_PFCs = data['PFCs']
    Ip = data['Ip']
    
    model.eval()
    psi_p = model(x_param.to(device), x_PFCs.to(device))
    psi_p_np = psi_p.detach().cpu().squeeze(0).numpy()
    
    # positional information
    R = model.R2D
    Z = model.Z2D
    
    if type(psi) == torch.Tensor:
        psi = psi.detach().cpu().numpy()
        
    fig, ax = plt.subplots(2,1,figsize=(4,8))
    ax[0].contourf(R,Z, psi, levels = 32)
    ax[0] = draw_KSTAR_limiter(ax[0])
    bndy = data['rzbdys'].cpu().squeeze(0).numpy()
    ax[0].plot(bndy[:,0], bndy[:,1], c = 'r', linewidth = 1.5)
    
    ax[1].contourf(R,Z,psi_p_np, levels = 32)
    ax[1] = draw_KSTAR_limiter(ax[1])
    
    if contour_regressor is not None:
        contour_regressor.eval()
        bndy = contour_regressor.compute_rzbdys(psi_p, x_param.to(device), x_PFCs.to(device))
        ax[1].plot(bndy[:,0], bndy[:,1], c = 'r', linewidth = 1.5)
    
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
    plt.savefig(os.path.join(save_dir, "{}_comparison.png".format(tag)))