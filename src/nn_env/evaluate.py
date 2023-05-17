import torch
import numpy as np
from torch.utils.data import DataLoader
from src.nn_env.metric import compute_metrics
from src.nn_env.predict import multi_step_prediction

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

    for batch_idx, (data_0D, data_ctrl, target) in enumerate(test_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            output = model(data_0D.to(device), data_ctrl.to(device))
            loss = loss_fn(output, target.to(device))
            test_loss += loss.item()
            
            pts.append(output.cpu().numpy().reshape(-1, output.size()[-1]))
            gts.append(target.cpu().numpy().reshape(-1, target.size()[-1]))
            
    test_loss /= (batch_idx + 1)
    
    pts = np.concatenate(pts, axis = 0)
    gts = np.concatenate(gts, axis = 0)
    
    mse, rmse, mae, r2 = compute_metrics(gts,pts,None,is_print)

    return test_loss, mse, rmse, mae, r2

# this version of evaluation process considers the multi-step prediction performance
# batch unit evaluation process for multi-step prediction
def evaluate_multi_step(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str,
    is_print : bool = True,
    ):
    
    model.eval()
    model.to(device)
    test_loss = 0
    
    seq_len_0D = model.input_0D_seq_len
    pred_len_0D = model.output_0D_pred_len
    
    total_mse = 0
    total_rmse = 0
    total_mae = 0
    total_r2 = 0
    
    for batch_idx, (data_0D, data_ctrl, target) in enumerate(test_loader):        
        with torch.no_grad():
            optimizer.zero_grad()
            preds = multi_step_prediction(model, data_0D, data_ctrl, seq_len_0D, pred_len_0D)
            preds = torch.from_numpy(preds)
            loss = loss_fn(preds.to(device), target.to(device))
            test_loss += loss.item()
            
            gts = target.numpy()
            pts = preds.numpy()
            mse, rmse, mae, r2 = compute_metrics(gts,pts,None, False)
            
            total_rmse += rmse
            total_mse += mse
            total_mae += mae
            total_r2 += r2
        
    test_loss /= batch_idx + 1
    total_rmse /= batch_idx + 1
    total_mse /= batch_idx + 1
    total_mae /= batch_idx + 1
    total_r2 /= batch_idx + 1
    
    if is_print:
        print("| mse : {:.3f} | rmse : {:.3f} | mae : {:.3f} | r2-score : {:.3f}".format(total_mse, total_rmse, total_mae, total_r2))
    
    return test_loss, total_mse, total_rmse, total_mae, total_r2