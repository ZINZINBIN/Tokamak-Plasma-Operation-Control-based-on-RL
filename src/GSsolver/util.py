import torch
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.pyplot import Axes
from src.GSsolver.KSTAR_setup import limiter_shape
from src.GSsolver.model import AbstractPINN
from matplotlib.gridspec import GridSpec

# draw KSTAR limiter (External boundary)
def draw_KSTAR_limiter(ax:Axes):
    ax.plot(limiter_shape[:,0], limiter_shape[:,1], 'black')
    ax.set_xlim([1.0, 2.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel('R[m]')
    ax.set_ylabel('Z[m]')
    return ax

# plot predicted flux, 2D current profile and 1D profile of pressure, current and safety-factor
def plot_profile(model : AbstractPINN, x_param : torch.Tensor, x_PFCs : torch.Tensor, device : str = "cpu", save_dir : str = "./result/PINN_profile.png"):
    
    model.eval()
    psi_p = model(x_param.to(device), x_PFCs.to(device))
    psi_p_np = psi_p.detach().cpu().squeeze(0).numpy()
    
    R = model.R2D
    Z = model.Z2D
    
    # toroidal current profile
    Jphi = model.compute_Jphi(psi_p)
    Jphi = Jphi.detach().cpu().squeeze(0).numpy()
    
    x_psi, ffprime = model.compute_ffprime()
    x_psi, pprime = model.compute_pprime()
    
    x_psi, pressure = model.compute_pressure_psi()    
    x_psi, Jphi1D = model.compute_Jphi_psi(psi_p)
    
    fig = plt.figure(figsize = (16, 5))
    fig.suptitle("PINN test result : $\Psi$, $J_\phi$, $P(\psi)$ profile")
    gs = GridSpec(nrows = 2, ncols = 4)
    
    ax = fig.add_subplot(gs[:,0])
    ax.contourf(R,Z,psi_p_np)
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
    
    ax = fig.add_subplot(gs[1,3])
    ax.plot(x_psi, pressure, 'r-', label = '$P(\psi)$')
    ax.set_xlabel("Normalized $\psi$")
    ax.set_ylabel("Relative value")
    ax.legend()

    fig.tight_layout()
    plt.savefig(save_dir)