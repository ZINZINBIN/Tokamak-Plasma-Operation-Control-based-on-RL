import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os, cv2, glob2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Literal, Union, List, Dict
from src.config import Config

config = Config()

def generate_control_performance(
        file_path : str,
        total_state : np.array,
        total_action : np.array,
        cols_0D : List,
        cols_ctrl : List,
        targets_dict : Dict,
        title : str,
        dt : float,
        plot_freq : int,
    ):
    
    # generate gif file using animation
    time_x = [i * dt for i in range(0, len(total_state))]
    
    # step 1. 0D case
    fig, axes = plt.subplots(len(cols_0D), 1, figsize = (16,12), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    fig.tight_layout()
    
    axes_info_0D = []
    
    for col, ax in zip(cols_0D, axes.ravel()):
        ax_point = ax.plot([],[], label = config.COL2STR[col])[0]
        axes_info_0D.append(ax_point)
    
    def replay(idx : int):
        for j, (ax_ori, ax) in enumerate(zip(axes.ravel(), axes_info_0D)):
            ax.set_data(time_x[:idx], total_state[:idx,j])
            
            col = cols_0D[j]
            
            if col in list(targets_dict.keys()):
                ax_ori.axhline(targets_dict[col], xmin = 0, xmax = 1)
                ax_ori.set_ylabel(config.COL2STR[col])
                ax_ori.legend(loc = "upper right")
            else:
                ax_ori.set_ylabel(config.COL2STR[col])
                ax_ori.legend(loc = "upper right") 
                
            ax_ori.set_xlabel('time')
            ax_ori.set_ylim([0,5])
            ax_ori.set_xlim([0, max(time_x)])
      
    indices = [i for i in range(len(total_state))]
    ani = animation.FuncAnimation(fig, replay, frames = indices)
    writergif = animation.PillowWriter(fps = plot_freq)
    ani.save(file_path, writergif)