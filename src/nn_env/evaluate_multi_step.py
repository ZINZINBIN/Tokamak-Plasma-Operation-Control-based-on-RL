import torch
import numpy as np
from torch.utils.data import DataLoader
from src.nn_env.metric import compute_metrics

def evaluate(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    is_print : bool = True,
    ):

    model.eval()
    model.to(device)
    test_loss = 0
    
    pts = []
    gts = []

    for batch_idx, (data_0D, data_ctrl, target_0D, target_ctrl, label) in enumerate(test_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            output = model(data_0D.to(device), data_ctrl.to(device), target_0D.to(device), target_ctrl.to(device))
            loss = loss_fn(output, label.to(device))
            test_loss += loss.item()
            
            pts.append(output.cpu().numpy().reshape(-1, output.size()[-1]))
            gts.append(label.cpu().numpy().reshape(-1, label.size()[-1]))
            
    test_loss /= (batch_idx + 1)
    
    pts = np.concatenate(pts, axis = 0)
    gts = np.concatenate(gts, axis = 0)
    
    mse, rmse, mae, r2 = compute_metrics(gts,pts,None,is_print)

    return test_loss, mse, rmse, mae, r2