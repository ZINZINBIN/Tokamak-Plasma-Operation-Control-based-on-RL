import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors, cm
from matplotlib.gridspec import GridSpec
from skimage import measure
import numpy as np
from typing import Optional, Literal, Union, List, Dict
from src.config import Config
from src.GSsolver.util import draw_KSTAR_limiter
from src.rl.env import NeuralEnv

config = Config()

def generate_control_performance(
        file_path : str,
        total_state : np.array,
        env : NeuralEnv,
        cols_0D : List,
        targets_dict : Dict,
        title : str,
        dt : float,
        plot_freq : int,
        seq_len : int,
        plot_boundary : bool = False,
        cols_plot : Optional[List] = None,
    ):
    
    total_flux = env.flux_list
    total_contour = env.contour_list
    total_axis = env.axis_list
    
    R = env.shape_predictor.R2D
    Z = env.shape_predictor.Z2D
    
    total_state = total_state[-len(total_flux):]
    total_xpts = env.xpts
    total_opts = env.opts
    
    # generate gif file using animation
    time_x = [i * dt for i in range(0, len(total_state))]
    
    fig = plt.figure(figsize = (14,9), facecolor="white")
    gs = GridSpec(nrows = len(cols_0D), ncols = 2)
    
    if cols_plot is not None:
        gs = GridSpec(nrows = len(cols_plot), ncols = 2)
    
    # generate axis for plotting each column of data
    axes_0D = []
    
    n_axes = len(cols_0D) if cols_plot is None else len(cols_plot)
    
    for idx in range(n_axes):
        ax = fig.add_subplot(gs[idx,0])
        axes_0D.append(ax)
        
    ax_flux = fig.add_subplot(gs[:,1])
    ax_flux.set_title("PINN test result : $\Psi$, $J_\phi$, $P(\psi)$ profile")
    
    psi_init = total_flux[0]
    norm = colors.Normalize(vmin = psi_init.min(), vmax = psi_init.max())
    map = cm.ScalarMappable(norm=norm)
    fig.colorbar(map, ax = ax_flux)

    plt.suptitle(title)
    fig.tight_layout()
    
    axes_0D_point = []
    
    for col, ax in zip(cols_0D, axes_0D):
        ax_point = ax.plot([],[], 'k', label = config.COL2STR[col])[0]
        axes_0D_point.append(ax_point)
        
    t_control = seq_len * dt
    
    def _plot(idx : int, axes_0D, axes_0D_point, ax_flux):
        # plot the 0D states of plasma
        for j, (ax_ori, ax) in enumerate(zip(axes_0D, axes_0D_point)):
            
            if cols_plot is not None:
                j = cols_0D.index(cols_plot[j])
            
            ax_ori.set_facecolor(plt.cm.Blues(0.2))
            ax.set_data(time_x[:idx], total_state[:idx,j])
            
            col = cols_0D[j]
            
            ax_ori.axvline(t_control, ymin = 0, ymax = 1, linewidth = 2, color = 'b')
            
            if col in list(targets_dict.keys()):
                ax_ori.axhline(targets_dict[col], xmin = 0, xmax = 1, linewidth = 4, color = 'y')
                ax_ori.set_ylabel(config.COL2STR[col])
                ax_ori.legend(loc = "upper right")
            else:
                ax_ori.set_ylabel(config.COL2STR[col])
                ax_ori.legend(loc = "upper right") 
                
            ax_ori.set_xlabel('time')
            ymin = min(total_state[:,j])
            ymax = max(total_state[:,j])
            
            if ymin < 0:
                ymin = ymin * 1.25
            else:
                ymin = ymin * 0.75
            
            if ymax < 0:
                ymax = ymax * 0.75
            else:
                ymax = ymax * 1.25
            
            ax_ori.set_ylim([ymin ,ymax])
            ax_ori.set_xlim([0, max(time_x)])

        # 2 contour the flux
        psi = total_flux[idx]
        
        ax_flux.clear()
        ax_flux.contourf(R,Z, psi, levels = 32)
        draw_KSTAR_limiter(ax_flux)
        
        ax_flux.set_xlabel("R[m]")
        ax_flux.set_ylabel("Z[m]")
        ax_flux.set_title('Poloidal flux ($\psi$)')
        
        if plot_boundary:
            
            '''
            r_axis, z_axis = total_opts[idx]
            xpts = total_xpts[idx]
            
            if r_axis is not None:
                ax_flux.plot(r_axis, z_axis, "o", c = "r", label = "magnetic axis", linewidth = 2)
            
            if len(xpts) > 0:
                r_xpts = []
                z_xpts = []
                psi_xpts = []
                
                for r_xpt, z_xpt, psi_xpt in xpts:
                    r_xpts.append(r_xpt)
                    z_xpts.append(z_xpt)
                    psi_xpts.append(psi_xpt)
                
                r_xpts = np.array(r_xpts)
                z_xpts = np.array(z_xpts)
                psi_xpts = np.array(psi_xpts)
                psi_b = np.min(psi_xpts)
            
            else:
                psi_b = 0.1
                
            try:
                if len(xpts) > 0:
                    contours = measure.find_contours(psi, psi_b)
                    dist_list = []
                    for contour in contours:
                        r_contour = R.min() + (R.max() - R.min()) * contour[:,1] / R.shape[0]
                        z_contour = Z.min() + (Z.max() - Z.min()) * contour[:,0] / Z.shape[0]
                        dist = np.mean((r_contour-r_axis) ** 2 + (z_contour - z_axis) ** 2)
                        dist_list.append(dist)
                        
                    b_contour = contours[np.argmin(np.array(dist_list))]
                    
                else:
                    b_contour = None
            except:
                b_contour = None
                        
            if b_contour is not None:
                r_contour = R.min() + (R.max() - R.min()) * b_contour[:,1] / R.shape[0]
                z_contour = Z.min() + (Z.max() - Z.min()) * b_contour[:,0] / Z.shape[0]
                ax_flux.plot(r_contour, z_contour, c = 'r', linewidth = 2) 
            ''' 
            
            # new version : use contour regressor
            axis = total_axis[idx]
            ax_flux.plot(axis[0], axis[1], "o", c = "r", label = "magnetic axis", linewidth = 2)
            ax_flux.legend(loc = 'upper right')
            
            contour = total_contour[idx]
            ax_flux.plot(contour[:,0], contour[:,1], c = 'r', linewidth = 2)
                
        fig.tight_layout()
  
    replay = lambda idx : _plot(idx, axes_0D, axes_0D_point, ax_flux)
      
    indices = [i for i in range(len(total_state))]
    ani = animation.FuncAnimation(fig, replay, frames = indices)
    writergif = animation.PillowWriter(fps = plot_freq)
    ani.save(file_path, writergif)