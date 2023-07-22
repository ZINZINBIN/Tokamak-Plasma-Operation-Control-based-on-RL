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
    model : AbstractPINN,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    device : str = "cpu",
    max_norm_grad : Optional[float] = 1.0,
    weights : Optional[Dict] = None,
    contour_regressor : Optional[ContourRegressor] = None,
    contour_optimizer : Optional[torch.optim.Optimizer] = None,
    contour_scheduler : Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
    
    model.train()
    model.to(device)
    
    train_loss = 0
    gs_loss = 0
    constraint_loss = 0
    ssim_loss = 0
    total_size = 0
    
    ip_constraint_loss = 0
    betap_constraint_loss = 0
    
    # loss defined
    loss_mse = nn.MSELoss(reduction = 'sum')
    loss_ssim = SSIM()
    
    if contour_regressor is not None:
        contour_regressor.train()
        contour_regressor.to(device)
        contour_loss_mse = nn.MSELoss(reduction = 'sum')
        train_loss_contour = 0
    else:
        contour_loss_mse = None
        train_loss_contour = None
    
    # weights
    if weights is None:
        weights = {
            "GS_loss" : 1.0,
            "Constraint_loss" : 1.0 
        }
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data['params'].to(device), data['PFCs'].to(device))
        loss = loss_mse(output, target.to(device))
        
        if getattr(model, 'compute_GS_loss'):
            gs_loss = model.compute_GS_loss(output) 
            
            if not torch.isnan(gs_loss):
                loss += gs_loss * weights['GS_loss']
                
            gs_loss = gs_loss.detach().cpu().item()
        
        if getattr(model, "compute_constraint_loss"):
            constraint_loss = model.compute_constraint_loss(output, data['Ip'].to(device), data['betap'].to(device)) 
            
            if not torch.isnan(constraint_loss):
                loss += constraint_loss * weights['Constraint_loss']
                
                ip_constraint_loss += model.compute_constraint_loss_Ip(output, data['Ip'].to(device)).detach().cpu().item()
                betap_constraint_loss += model.compute_constraint_loss_betap(output, data['Ip'].to(device), data['betap'].to(device)).detach().cpu().item()
                
            constraint_loss = constraint_loss.detach().cpu().item()

        # backward process
        if not torch.isnan(loss):
            loss.backward()
    
            if max_norm_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)
                
            optimizer.step()

            train_loss += loss.detach().cpu().item()
            ssim_loss += loss_ssim(output.detach(), target.to(device)).detach().cpu().item()
            total_size += target.size()[0]
        
        # parameter range fixed
        with torch.no_grad():
            model.lamda.clamp_(0.1, 10)
            model.beta.clamp_(0.1, 0.9)
            
        if contour_regressor is not None:
            contour_optimizer.zero_grad()
            output = contour_regressor(target.to(device))
            contour_loss = contour_loss_mse(output, data['rzbdys'].to(device))
            
            if not torch.isnan(contour_loss):
                contour_loss.backward()
                contour_optimizer.step()
                train_loss_contour += contour_loss.detach().cpu().item()
            else:
                pass
            
    if scheduler:
        scheduler.step()
        
    if contour_scheduler:
        contour_scheduler.step()
        
    if total_size > 0:
        train_loss /= total_size
        gs_loss /= total_size
        constraint_loss /= total_size
        ssim_loss /= total_size
        
        ip_constraint_loss /= total_size
        betap_constraint_loss /= total_size
        
        if train_loss_contour is not None:
            train_loss_contour /= total_size
        
    else:
        train_loss = 0
        gs_loss = 0
        constraint_loss = 0
        ssim_loss = 0
        
        ip_constraint_loss = 0
        betap_constraint_loss = 0
    
    return train_loss, gs_loss, constraint_loss, ip_constraint_loss, betap_constraint_loss, ssim_loss, train_loss_contour
        
        
def valid_per_epoch(
    dataloader : DataLoader,
    model : AbstractPINN,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    device : str = "cpu",
    weights : Optional[Dict] = None,
    contour_regressor : Optional[ContourRegressor] = None,
    contour_optimizer : Optional[torch.optim.Optimizer] = None,
    contour_scheduler : Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
    
    model.eval()
    model.to(device)
    
    valid_loss = 0
    gs_loss = 0
    constraint_loss = 0
    ssim_loss = 0
    total_size = 0
    
    ip_constraint_loss = 0
    betap_constraint_loss = 0
    
    # loss defined
    loss_mse = nn.MSELoss(reduction = 'sum')
    loss_ssim = SSIM()
    
    # weights
    if weights is None:
        weights = {
            "GS_loss" : 1.0,
            "Constraint_loss" : 1.0 
        }
        
    if contour_regressor is not None:
        contour_regressor.eval()
        contour_regressor.to(device)
        contour_loss_mse = nn.MSELoss(reduction = 'sum')
        valid_loss_contour = 0
    else:
        contour_loss_mse = None
        valid_loss_contour = None
    
    for batch_idx, (data, target) in enumerate(dataloader):
        
        optimizer.zero_grad()
        
        output = model(data['params'].to(device), data['PFCs'].to(device))
        loss = loss_mse(output, target.to(device))
            
        if getattr(model, 'compute_GS_loss'):
            loss += model.compute_GS_loss(output) * weights['GS_loss']
            gs_loss += model.compute_GS_loss(output).detach().cpu().item()
        
        if getattr(model, "compute_constraint_loss"):
            loss += model.compute_constraint_loss(output, data['Ip'].to(device), data['betap'].to(device)) * weights['Constraint_loss']
            constraint_loss += model.compute_constraint_loss(output, data['Ip'].to(device), data['betap'].to(device)).detach().cpu().item()
            
            ip_constraint_loss += model.compute_constraint_loss_Ip(output, data['Ip'].to(device)).detach().cpu().item()
            betap_constraint_loss += model.compute_constraint_loss_betap(output, data['Ip'].to(device), data['betap'].to(device)).detach().cpu().item()
                

        valid_loss += loss.detach().cpu().item()
        ssim_loss += loss_ssim(output.detach(), target.to(device)).detach().cpu().item()
        total_size += target.size()[0]
        
        if contour_regressor is not None:
            with torch.no_grad():
                contour_optimizer.zero_grad()
                output = contour_regressor(target.to(device))
                contour_loss = contour_loss_mse(output, data['rzbdys'].to(device))
            
            if not torch.isnan(contour_loss):
                valid_loss_contour += contour_loss.detach().cpu().item()
            else:
                pass
    
    if total_size > 0:
        valid_loss /= total_size
        gs_loss /= total_size
        constraint_loss /= total_size
        ssim_loss /= total_size
        
        ip_constraint_loss /= total_size
        betap_constraint_loss /= total_size
        
        if valid_loss_contour is not None:
            valid_loss_contour /= total_size
    
    else:
        valid_loss = 0
        gs_loss = 0
        constraint_loss = 0
        ssim_loss = 0

        ip_constraint_loss = 0
        betap_constraint_loss = 0
    
    return valid_loss, gs_loss, constraint_loss, ip_constraint_loss, betap_constraint_loss, ssim_loss, valid_loss_contour
        
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
    test_for_check : Optional[DataLoader] = None,
    contour_regressor : Optional[ContourRegressor] = None,
    contour_optimizer : Optional[torch.optim.Optimizer] = None,
    contour_scheduler : Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    contour_save_best_dir : Optional[str] = "./weights/best.pt",
    contour_save_last_dir : Optional[str] = "./weights/last.pt",
    ):

    train_loss_list = []
    valid_loss_list = []
    
    best_epoch = 0
    best_loss = np.inf
    
    if contour_regressor is not None:
        best_loss_contour = np.inf
    else:
        best_loss_contour = None
    
    for epoch in tqdm(range(num_epoch), desc = 'training process'):
        
        train_loss, train_gs_loss, train_constraint_loss, train_ip_constraint_loss, train_betap_constraint_loss, train_ssim_loss, train_loss_contour = train_per_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            device,
            max_norm_grad,
            weights,
            contour_regressor,
            contour_optimizer,
            contour_scheduler
        )
        
        valid_loss, valid_gs_loss, valid_constraint_loss, valid_ip_constraint_loss, valid_betap_constraint_loss, valid_ssim_loss, valid_loss_contour = valid_per_epoch(
            valid_loader,
            model,
            optimizer,
            scheduler,
            device,
            weights,
            contour_regressor,
            contour_optimizer,
            contour_scheduler
        )
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        if epoch % verbose == 0:
            if train_loss_contour is not None:
                print("Epoch:{} | train loss:{:.3f} | GS loss:{:.3f} | Constraint(Ip):{:.3f} | Constraint(betap):{:.3f} | SSIM:{:.3f} | contour loss:{:.3f}".format(epoch+1, train_loss, train_gs_loss, train_ip_constraint_loss, train_betap_constraint_loss,train_ssim_loss,train_loss_contour))
                print("Epoch:{} | valid loss:{:.3f} | GS loss:{:.3f} | Constraint(Ip):{:.3f} | Constraint(betap):{:.3f} | SSIM:{:.3f} | contour loss:{:.3f}".format(epoch+1, valid_loss, valid_gs_loss, valid_ip_constraint_loss, valid_betap_constraint_loss,valid_ssim_loss,valid_loss_contour))
            else:
                print("Epoch:{} | train loss:{:.3f} | GS loss:{:.3f} | Constraint(Ip):{:.3f} | Constraint(betap):{:.3f} | SSIM:{:.3f}".format(epoch+1, train_loss, train_gs_loss, train_ip_constraint_loss, train_betap_constraint_loss,train_ssim_loss))
                print("Epoch:{} | valid loss:{:.3f} | GS loss:{:.3f} | Constraint(Ip):{:.3f} | Constraint(betap):{:.3f} | SSIM:{:.3f}".format(epoch+1, valid_loss, valid_gs_loss, valid_ip_constraint_loss, valid_betap_constraint_loss, valid_ssim_loss))
        
        torch.save(model.state_dict(), save_last_dir)   

        if valid_loss < best_loss:
            best_epoch = epoch
            best_loss = valid_loss
            torch.save(model.state_dict(), save_best_dir)
            
        if contour_regressor is not None:
            torch.save(contour_regressor.state_dict(), contour_save_last_dir)
            
            if valid_loss_contour < best_loss_contour:
                best_loss_contour = valid_loss_contour
                torch.save(contour_regressor.state_dict(), contour_save_best_dir)
    
    print("Training process finished, best loss:{:.3f}, best epoch : {}".format(best_loss, best_epoch))    
    
    return train_loss_list, valid_loss_list