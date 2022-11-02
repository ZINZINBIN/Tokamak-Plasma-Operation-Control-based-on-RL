import torch
from torch.utils.data import DataLoader

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

    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            loss = loss_fn(output, target)
    
            test_loss += loss.item()

    test_loss /= (batch_idx + 1)
    print("test loss : {:.3f}".format(test_loss))

    return test_loss