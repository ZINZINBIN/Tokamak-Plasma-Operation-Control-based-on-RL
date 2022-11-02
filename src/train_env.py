from typing import Optional, List, Literal, Union
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

def train_per_epoch(
    train_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None
    ):

    model.train()
    model.to(device)

    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
    
        output = model(data)
        
        loss = loss_fn(output, target)

        loss.backward()

        # use gradient clipping
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

        optimizer.step()

        train_loss += loss.item()

    if scheduler:
        scheduler.step()

    train_loss /= (batch_idx + 1)

    return train_loss

def valid_per_epoch(
    valid_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)
    valid_loss = 0

    for batch_idx, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            loss = loss_fn(output, target)
    
            valid_loss += loss.item()

    valid_loss /= (batch_idx + 1)

    return valid_loss

def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best : str = "./weights/best.pt",
    save_last : str = "./weights/last.pt",
    max_norm_grad : Optional[float] = None,
    ):

    train_loss_list = []
    valid_loss_list = []

    best_epoch = 0
    best_loss = torch.inf

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad
        )

        valid_loss = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device 
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f},".format(
                    epoch+1, train_loss, valid_loss
                ))

        import os

        # save the best parameters
        if best_loss > valid_loss:
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best)

        # save the last parameters
        torch.save(model.state_dict(), save_last)

    # print("\n============ Report ==============\n")
    print("training process finished, best loss : {:.3f}, best epoch : {}".format(
        best_loss, best_epoch
    ))

    return  train_loss_list, valid_loss_list