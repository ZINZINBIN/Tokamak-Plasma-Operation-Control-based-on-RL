import numpy as np
import torch
import torch.nn as nn
# from numba import jit
from typing import Optional
from torch.autograd import Function 

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction = 'none')
    
    def forward(self, pred : torch.Tensor, target : torch.Tensor):
        loss = torch.sqrt(self.mse_loss(pred, target).mean(dim = 0)) / target.abs().mean(dim = 0)
        loss = loss.view(-1,).sum()
        return loss

# the code below is Dynamic Time Warping method aimed for time series forecasting
'''
# Soft Dynamic Time Warping
def pairwise_distances(x, y=None):

    # Input: x is a Nxd matrix
    #        y is an optional Mxd matirx
    # Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    #         if y is not given then use 'y=x'.
    # i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))

@jit(nopython = True)
def compute_softdtw(D, gamma):
    N = D.shape[0]
    M = D.shape[1]
    R = np.zeros((N + 2, M + 2)) + 1e8
    R[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            rmax = max(max(r0, r1), r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            softmin = - gamma * (np.log(rsum) + rmax)
            R[i, j] = D[i - 1, j - 1] + softmin
    return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
    N = D_.shape[0]
    M = D_.shape[1]
    D = np.zeros((N + 2, M + 2))
    E = np.zeros((N + 2, M + 2))
    D[1:N + 1, 1:M + 1] = D_
    E[-1, -1] = 1
    R[:, -1] = -1e8
    R[-1, :] = -1e8
    R[-1, -1] = R[-2, -2]
    
    for j in range(M, 0, -1):
        for i in range(N, 0, -1):
            a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
            b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
            c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
            a = np.exp(a0)
            b = np.exp(b0)
            c = np.exp(c0)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
    
    return E[1:N + 1, 1:M + 1]
 

class SoftDTWBatch(Function):
    @staticmethod
    def forward(ctx, D : torch.Tensor, gamma : float = 1.0): # D.shape: [batch_size, N , N]
        dev = D.device
        batch_size,N,N = D.shape
        gamma = torch.FloatTensor([gamma]).to(dev)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        total_loss = 0
        R = torch.zeros((batch_size, N+2 ,N+2)).to(dev)   
        for k in range(0, batch_size): # loop over all D in the batch    
            Rk = torch.FloatTensor(compute_softdtw(D_[k,:,:], g_)).to(dev)
            R[k:k+1,:,:] = Rk
            total_loss = total_loss + Rk[-2,-2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size
  
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size,N,N = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()

        E = torch.zeros((batch_size, N ,N)).to(dev) 
        for k in range(batch_size):         
            Ek = torch.FloatTensor(compute_softdtw_backward(D_[k,:,:], R_[k,:,:], g_)).to(dev)
            E[k:k+1,:,:] = Ek

        return grad_output * E, None
        
@jit(nopython = True)
def my_max(x, gamma):
    # use the log-sum-exp trick
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z

@jit(nopython = True)
def my_min(x,gamma) :
    min_x, argmax_x = my_max(-x, gamma)
    return - min_x, argmax_x

@jit(nopython = True)
def my_max_hessian_product(p, z, gamma):
    return  ( p * z - p * np.sum(p * z) ) /gamma

@jit(nopython = True)
def my_min_hessian_product(p, z, gamma):
    return - my_max_hessian_product(p, z, gamma)

@jit(nopython = True)
def dtw_grad(theta, gamma):
    m = theta.shape[0]
    n = theta.shape[1]
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            v, Q[i, j] = my_min(np.array([V[i, j - 1],V[i - 1, j - 1],V[i - 1, j]]) , gamma)                        
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    Q[m + 1, n + 1] = 1

    for i in range(m,0,-1):
        for j in range(n,0,-1):
            E[i, j] = Q[i, j + 1, 0] * E[i, j + 1] + \
                      Q[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                      Q[i + 1, j, 2] * E[i + 1, j]
    
    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


@jit(nopython = True)
def dtw_hessian_prod(theta, Z, Q, E, gamma):
    m = Z.shape[0]
    n = Z.shape[1]

    V_dot = np.zeros((m + 1, n + 1))
    V_dot[0, 0] = 0

    Q_dot = np.zeros((m + 2, n + 2, 3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            V_dot[i, j] = Z[i - 1, j - 1] + \
                          Q[i, j, 0] * V_dot[i, j - 1] + \
                          Q[i, j, 1] * V_dot[i - 1, j - 1] + \
                          Q[i, j, 2] * V_dot[i - 1, j]

            v = np.array([V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]])
            Q_dot[i, j] = my_min_hessian_product(Q[i, j], v, gamma)
    E_dot = np.zeros((m + 2, n + 2))

    for j in range(n,0,-1):
        for i in range(m,0,-1):
            E_dot[i, j] = Q_dot[i, j + 1, 0] * E[i, j + 1] + \
                          Q[i, j + 1, 0] * E_dot[i, j + 1] + \
                          Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                          Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1] + \
                          Q_dot[i + 1, j, 2] * E[i + 1, j] + \
                          Q[i + 1, j, 2] * E_dot[i + 1, j]

    return V_dot[m, n], E_dot[1:m + 1, 1:n + 1]


class PathDTWBatch(Function):
    @staticmethod
    def forward(ctx, D : torch.Tensor, gamma:float): # D.shape: [batch_size, N , N]
        batch_size,N,N = D.shape
        device = D.device
        D_cpu = D.detach().cpu().numpy()
        gamma_gpu = torch.FloatTensor([gamma]).to(device)
        
        grad_gpu = torch.zeros((batch_size, N ,N)).to(device)
        Q_gpu = torch.zeros((batch_size, N+2 ,N+2,3)).to(device)
        E_gpu = torch.zeros((batch_size, N+2 ,N+2)).to(device)  
        
        for k in range(0,batch_size): # loop over all D in the batch    
            _, grad_cpu_k, Q_cpu_k, E_cpu_k = dtw_grad(D_cpu[k,:,:], gamma)     
            grad_gpu[k,:,:] = torch.FloatTensor(grad_cpu_k).to(device)
            Q_gpu[k,:,:,:] = torch.FloatTensor(Q_cpu_k).to(device)
            E_gpu[k,:,:] = torch.FloatTensor(E_cpu_k).to(device)
        ctx.save_for_backward(grad_gpu,D, Q_gpu ,E_gpu, gamma_gpu) 
        return torch.mean(grad_gpu, dim=0) 
    
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        device = grad_output.device
        grad_gpu, D_gpu, Q_gpu, E_gpu, gamma = ctx.saved_tensors
        D_cpu = D_gpu.detach().cpu().numpy()
        Q_cpu = Q_gpu.detach().cpu().numpy()
        E_cpu = E_gpu.detach().cpu().numpy()
        gamma = gamma.detach().cpu().numpy()[0]
        Z = grad_output.detach().cpu().numpy()
        
        batch_size,N,N = D_cpu.shape
        Hessian = torch.zeros((batch_size, N ,N)).to(device)
        for k in range(0,batch_size):
            _, hess_k = dtw_hessian_prod(D_cpu[k,:,:], Z, Q_cpu[k,:,:,:], E_cpu[k,:,:], gamma)
            Hessian[k:k+1,:,:] = torch.FloatTensor(hess_k).to(device)

        return  Hessian, None
    
    
def dilate_loss(outputs:torch.Tensor, targets:torch.Tensor, alpha, gamma, device):
    batch_size, seq_len, feat_dim = outputs.size()
    loss_shape = 0
    D = torch.zeros((batch_size, seq_len, seq_len)).to(device)
    for k in range(batch_size):
        Dk = torch.zeros((seq_len, seq_len)).to(device)
        for j in range(feat_dim):
            Dk += pairwise_distances(targets[k,:,j].view(-1,1),outputs[k,:,j].view(-1,1))
        D[k,:,:] = Dk   
          
    loss_shape = SoftDTWBatch.apply(D,gamma)
    path = PathDTWBatch.apply(D, gamma)           
    Omega =  pairwise_distances(torch.arange(seq_len).view(seq_len, 1)).to(device)
    loss_temporal =  torch.sum(path*Omega ) / (seq_len*seq_len) 
    loss = alpha*loss_shape+ (1-alpha)*loss_temporal
    return loss, loss_shape, loss_temporal

class DILATE(nn.Module):
    def __init__(self, alpha : float = 0.5, gamma : float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction = 'mean')
    
    def forward(self, pts : torch.Tensor, gts : torch.Tensor):
        loss, _, _ = dilate_loss(pts, gts, self.alpha, self.gamma, pts.device)
        mse_loss = self.mse(pts, gts)
    
        return mse_loss + loss * 0.1
'''