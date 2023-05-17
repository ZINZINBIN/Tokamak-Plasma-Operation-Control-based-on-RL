import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, pdb
from typing import Dict, Optional, List, Literal, Union
from src.GSsolver.model import AbstractPINN
from src.GSsolver.loss import SSIM

def train_per_epoch(
    dataloader : DataLoader,
    model : AbstractPINN,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    device : str = "cpu",
    max_norm_grad : Optional[float] = 1.0,
    weights : Optional[Dict] = None
    ):
    
    model.train()
    model.to(device)
    
    train_loss = 0
    gs_loss = 0
    constraint_loss = 0
    ssim_loss = 0
    total_size = 0
    
    # loss defined
    loss_mse = nn.MSELoss(reduction = 'sum')
    loss_ssim = SSIM()
    
    # weights
    if weights is None:
        weights = {
            "GS_loss" : 0.1,
            "Constraint_loss" : 0.1 
        }
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data['params'].to(device), data['PFCs'].to(device))
        
        loss = loss_mse(output, target.to(device))
        
        if getattr(model, 'compute_GS_loss'):
            loss += model.compute_GS_loss(output) * weights['GS_loss']
            gs_loss += model.compute_GS_loss(output).detach().cpu().item()
        
        if getattr(model, "compute_constraint_loss"):
            loss += model.compute_constraint_loss(output.detach(), data['Ip'].to(device)) * weights['Constraint_loss']
            constraint_loss += model.compute_constraint_loss(output.detach(), data['Ip'].to(device)).detach().cpu().item()

        loss.backward()
        
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)
        
        optimizer.step()

        train_loss += loss.detach().cpu().item()
        ssim_loss += loss_ssim(output.detach(), target.to(device)).detach().cpu().item()
        total_size += target.size()[0]
    
    if scheduler:
        scheduler.step()
        
    if total_size > 0:
        train_loss /= total_size
        gs_loss /= total_size
        constraint_loss /= total_size
        ssim_loss /= total_size
    else:
        train_loss = 0
        gs_loss = 0
        constraint_loss = 0
        ssim_loss = 0
    
    return train_loss, gs_loss, constraint_loss, ssim_loss
        
        
def valid_per_epoch(
    dataloader : DataLoader,
    model : AbstractPINN,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    device : str = "cpu",
    weights : Optional[Dict] = None
    ):
    
    model.eval()
    model.to(device)
    
    valid_loss = 0
    gs_loss = 0
    constraint_loss = 0
    ssim_loss = 0
    total_size = 0
    
    # loss defined
    loss_mse = nn.MSELoss(reduction = 'sum')
    loss_ssim = SSIM()
    
    # weights
    if weights is None:
        weights = {
            "GS_loss" : 0.1,
            "Constraint_loss" : 0.1 
        }
    
    for batch_idx, (data, target) in enumerate(dataloader):
        with torch.no_grad():
            output = model(data['params'].to(device), data['PFCs'].to(device))
            loss = loss_mse(output, target.to(device))
            
            if getattr(model, 'compute_GS_loss'):
                loss += model.compute_GS_loss(output) * weights['GS_loss']
                gs_loss += model.compute_GS_loss(output).detach().cpu().item()
            
            if getattr(model, "compute_constraint_loss"):
                loss += model.compute_constraint_loss(output.detach(), data['Ip'].to(device)) * weights['Constraint_loss']
                constraint_loss += model.compute_constraint_loss(output.detach(), data['Ip'].to(device)).detach().cpu().item()

            valid_loss += loss.detach().cpu().item()
            ssim_loss += loss_ssim(output.detach(), target.to(device)).detach().cpu().item()
            total_size += target.size()[0]
    
    if scheduler:
        scheduler.step()
        
    if total_size > 0:
        valid_loss /= total_size
        gs_loss /= total_size
        constraint_loss /= total_size
        ssim_loss /= total_size
    else:
        valid_loss = 0
        gs_loss = 0
        constraint_loss = 0
        ssim_loss = 0
    
    return valid_loss, gs_loss, constraint_loss, ssim_loss
        
def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : AbstractPINN,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_dir : str = "./weights/best.pt",
    save_last_dir : str = "./weights/last.pt",
    max_norm_grad : Optional[float] = None,
    weights : Optional[Dict] = None,
    ):
    
    train_loss_list = []
    valid_loss_list = []
    
    best_epoch = 0
    best_loss = np.inf
    
    for epoch in tqdm(range(num_epoch), desc = 'training process'):
        
        train_loss, train_gs_loss, train_constraint_loss, train_ssim_loss = train_per_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            device,
            max_norm_grad,
            weights
        )
        
        valid_loss, valid_gs_loss, valid_constraint_loss, valid_ssim_loss = valid_per_epoch(
            valid_loader,
            model,
            optimizer,
            scheduler,
            device,
            weights
        )
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        if verbose:
            print("Epoch:{} | train loss:{:.3f} | GS loss:{:.3f} | Constraint:{:.3f} | SSIM :{:.3f}".format(epoch+1, train_loss, train_gs_loss, train_constraint_loss, train_ssim_loss))
            print("Epoch:{} | valid loss:{:.3f} | GS loss:{:.3f} | Constraint:{:.3f} | SSIM :{:.3f}".format(epoch+1, valid_loss, valid_gs_loss, valid_constraint_loss, valid_ssim_loss))
        
        torch.save(model.state_dict(), save_last_dir)    
        
        if valid_loss < best_loss:
            best_epoch = epoch
            best_loss = valid_loss
            torch.save(model.state_dict(), save_best_dir)
    
    print("Training process finished, best loss : {:.3f}, best epoch : {}".format(best_loss, best_epoch))    
    
    return train_loss_list, valid_loss_list