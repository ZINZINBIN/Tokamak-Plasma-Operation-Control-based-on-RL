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
    ):

    model.eval()
    model.to(device)
    test_loss = 0
    
    pts = []
    gts = []

    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            loss = loss_fn(output, target)
    
            test_loss += loss.item()
            
            pts.append(output.cpu().numpy().reshape(-1, output.size()[-1]))
            gts.append(target.cpu().numpy().reshape(-1, target.size()[-1]))
            
    test_loss /= (batch_idx + 1)
    print("test loss : {:.3f}".format(test_loss))
    
    pts = np.concatenate(pts, axis = 0)
    gts = np.concatenate(gts, axis = 0)
    
    mse, rmse, mae = compute_metrics(gt,pt,None,True)

    return test_loss