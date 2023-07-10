import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors, cm
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Optional, Literal, Union, List, Dict
from src.config import Config
from src.GSsolver.util import draw_KSTAR_limiter

config = Config()

def generate_control_performance(
        file_path : str,
        total_state : np.array,
        total_flux : Optional[List],
        R,
        Z,
        cols_0D : List,
        targets_dict : Dict,
        title : str,
        dt : float,
        plot_freq : int,
        seq_len : int,
    ):
    
    total_state = total_state[-len(total_flux):]
    
    # generate gif file using animation
    time_x = [i * dt for i in range(0, len(total_state))]
    
    fig = plt.figure(figsize = (14,9), facecolor="white")
    gs = GridSpec(nrows = len(cols_0D), ncols = 2)
    
    # generate axis for plotting each column of data
    axes_0D = []
    
    for idx in range(len(cols_0D)):
        ax = fig.add_subplot(gs[idx,0])
        axes_0D.append(ax)
        
    ax_flux = fig.add_subplot(gs[:,1])
    ax_flux.set_title("PINN test result : $\Psi$, $J_\phi$, $P(\psi)$ profile")

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
        ax_flux.contourf(R,Z,psi,levels = 32)
        draw_KSTAR_limiter(ax_flux)
        '''
        if idx == 0:
            norm = colors.Normalize(vmin = psi.min(), vmax = psi.max())
            map = cm.ScalarMappable(norm=norm)
            fig.colorbar(map)
        '''
        ax_flux.set_xlabel("R[m]")
        ax_flux.set_ylabel("Z[m]")
        ax_flux.set_title('Poloidal flux ($\psi$)')
        
    replay = lambda idx : _plot(idx, axes_0D, axes_0D_point, ax_flux)
      
    indices = [i for i in range(len(total_state))]
    ani = animation.FuncAnimation(fig, replay, frames = indices)
    writergif = animation.PillowWriter(fps = plot_freq)
    ani.save(file_path, writergif)