import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from src.GSsolver.model import PINN, ContourRegressor
from src.GSsolver.loss import SSIM

def evaluate(
    dataloader : DataLoader,
    model : PINN,
    device : str = "cpu",
    weights : Optional[Dict] = None,
    contour_regressor : Optional[ContourRegressor] = None,
    ):
    
    model.eval()
    model.to(device)
    
    test_loss = 0
    gs_loss = 0
    constraint_loss = 0
    ip_constraint_loss = 0
    betap_constraint_loss = 0
    ssim_loss = 0
    total_size = 0
    
    if contour_regressor is not None:
        contour_regressor.eval()
        contour_regressor.to(device)
    
    # loss defined
    loss_mse = nn.MSELoss(reduction = 'sum')
    loss_ssim = SSIM()
    
    if contour_regressor is not None:
        test_contour_loss = 0
        contour_loss_mse = nn.MSELoss(reduction='sum')
    else:
        test_contour_loss = None
        contour_loss_mse = None
    
    # weights
    if weights is None:
        weights = {
            "GS_loss" : 1.0,
            "Constraint_loss" : 1.0 
        }
    
    for batch_idx, (data, target) in enumerate(dataloader):
        
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

        test_loss += loss.detach().cpu().item()
        ssim_loss += loss_ssim(output.detach(), target.to(device)).detach().cpu().item()
        total_size += target.size()[0]
        
        
        if contour_regressor is not None:
            with torch.no_grad():
                output = contour_regressor(target.to(device))
                contour_loss = contour_loss_mse(output, data['rzbdys'].to(device))
            
            if not torch.isnan(contour_loss):
                test_contour_loss += contour_loss.detach().cpu().item()
            else:
                pass
    
    if total_size > 0:
        test_loss /= total_size
        gs_loss /= total_size
        constraint_loss /= total_size
        ssim_loss /= total_size
        
        ip_constraint_loss /= total_size
        betap_constraint_loss /= total_size
        
        if test_contour_loss is not None:
            test_contour_loss /= total_size
    
    else:
        test_loss = 0
        gs_loss = 0
        constraint_loss = 0
        ssim_loss = 0
        
        ip_constraint_loss = 0
        betap_constraint_loss = 0
        
    if test_contour_loss is not None:
        print("Evaluation | test loss:{:.3f} | GS loss:{:.3f} | Constraint(Ip):{:.3f} | Constraint(betap):{:.3f} |SSIM :{:.3f} | contour loss:{:.3f}".format(test_loss, gs_loss, ip_constraint_loss, betap_constraint_loss, ssim_loss, test_contour_loss))
    else:
        print("Evaluation | test loss:{:.3f} | GS loss:{:.3f} | Constraint(Ip):{:.3f} | Constraint(betap):{:.3f} |SSIM :{:.3f}".format(test_loss, gs_loss, ip_constraint_loss, betap_constraint_loss, ssim_loss))
        
    return test_loss, gs_loss, constraint_loss, ssim_loss, test_contour_loss