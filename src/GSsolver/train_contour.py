import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, pdb
from typing import Dict, Optional, List, Literal, Union
from src.GSsolver.model import AbstractPINN, ContourRegressor
from src.GSsolver.loss import SSIM
from src.GSsolver.util import plot_PINN_profile

def train_per_epoch(
    dataloader : DataLoader,
    model : ContourRegressor,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    device : str = "cpu",
    max_norm_grad : Optional[float] = 1.0,
    weight : Optional[List] = [1.0, 5.0],
    ):
    
    model.train()
    model.to(device)
    
    train_loss = 0
    total_size = 0
    
    # loss defined
    loss_mse_c = nn.MSELoss(reduction = 'mean')
    loss_mse_r = nn.MSELoss(reduction = 'mean')
  
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        cen, rad = model(target.to(device), data['params'].to(device), data['PFCs'].to(device))
        loss = loss_mse_r(rad, data['rad'].to(device)) * weight[0] + loss_mse_c(cen, data['center'].to(device)) * weight[1]
        
        total_size += target.size()[0]
        
        if not torch.isnan(loss):
            loss.backward()
            
            if max_norm_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)
                
            optimizer.step()
            train_loss += loss.detach().cpu().item()
        else:
            pass
            
    if scheduler:
        scheduler.step()
        
    if total_size > 0:
        train_loss /= total_size
    else:
        train_loss = 0
    
    return train_loss
        
        
def valid_per_epoch(
    dataloader : DataLoader,
    model : ContourRegressor,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    device : str = "cpu",
    weight : Optional[List] = [1.0, 5.0],
    ):
    
    model.eval()
    model.to(device)
    
    valid_loss = 0
    total_size = 0
    
    # loss defined
    loss_mse_c = nn.MSELoss(reduction = 'mean')
    loss_mse_r = nn.MSELoss(reduction = 'mean')
  
    for batch_idx, (data, target) in enumerate(dataloader):
        with torch.no_grad():
            optimizer.zero_grad()
            cen, rad = model(target.to(device), data['params'].to(device), data['PFCs'].to(device))
            loss = loss_mse_r(rad, data['rad'].to(device)) * weight[0] + loss_mse_c(cen, data['center'].to(device)) * weight[1]
        
            total_size += target.size()[0]
        
            if not torch.isnan(loss):
                valid_loss += loss.detach().cpu().item()
            else:
                pass

    if total_size > 0:
        valid_loss /= total_size
    else:
        valid_loss = 0
    
    return valid_loss
        
def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : ContourRegressor,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_dir : str = "./weights/best.pt",
    save_last_dir : str = "./weights/last.pt",
    max_norm_grad : Optional[float] = None,
    ):

    train_loss_list = []
    valid_loss_list = []
    
    best_epoch = 0
    best_loss = np.inf
    
    for epoch in tqdm(range(num_epoch), desc = 'training process'):
        
        train_loss = train_per_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            device,
            max_norm_grad,
        )
        
        valid_loss = valid_per_epoch(
            valid_loader,
            model,
            optimizer,
            scheduler,
            device,
        )
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        if epoch % verbose == 0:
            print("Epoch:{} | train loss:{:.3f}".format(epoch+1, train_loss))
            print("Epoch:{} | valid loss:{:.3f}".format(epoch+1, valid_loss))
        
        torch.save(model.state_dict(), save_last_dir)   

        if valid_loss < best_loss:
            best_epoch = epoch
            best_loss = valid_loss
            torch.save(model.state_dict(), save_best_dir)
            
    print("Training process finished, best loss:{:.3f}, best epoch : {}".format(best_loss, best_epoch))    
    
    return train_loss_list, valid_loss_list

def evaluate(
    dataloader : DataLoader,
    model : ContourRegressor,
    device : str = "cpu",
    weight : Optional[List] = [1.0, 5.0],
    ):
    
    model.eval()
    model.to(device)
    
    test_loss = 0
    total_size = 0
    
    # loss defined
    loss_mse_c = nn.MSELoss(reduction = 'mean')
    loss_mse_r = nn.MSELoss(reduction = 'mean')
  
    for batch_idx, (data, target) in enumerate(dataloader):
        with torch.no_grad():
            cen, rad = model(target.to(device), data['params'].to(device), data['PFCs'].to(device))
            loss = loss_mse_r(rad, data['rad'].to(device)) * weight[0] + loss_mse_c(cen, data['center'].to(device)) * weight[1]
            
            total_size += target.size()[0]
        
            if not torch.isnan(loss):
                test_loss += loss.detach().cpu().item()
            else:
                pass

    if total_size > 0:
        test_loss /= total_size
    else:
        test_loss = 0
    print("Evaluation | test loss:{:.3f}".format(test_loss))

    return test_loss