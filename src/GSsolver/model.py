import torch
import torch.nn as nn
import math
import numpy as np
from scipy.integrate import romb, quad, simpson
from scipy.interpolate import interp1d, RectBivariateSpline
from src.GSsolver.GradShafranov import *
from skimage import measure
from src.GSsolver.KSTAR_setup import PF_coils

class AbstractPINN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        return NotImplementedError('PINN::forward()')
    
    def predict(self):
        return NotImplementedError('PINN::predict()')
    
    def compute_Jphi(self):
        return NotImplementedError('PINN::compute_Jphi()')
    
    def compute_plasma_region(self):
        return NotImplementedError('PINN::compute_plasma_region()')
    
    def compute_GS_loss(self):
        return NotImplementedError('PINN::compute_GS_loss()')
    
    def compute_constraint_loss(self):
        return NotImplementedError('PINN::compute_constraint_loss()')
    
    def compute_Br(self):
        return NotImplementedError('PINN::compute_Br()')
    
    def compute_Bz(self):
        return NotImplementedError('PINN::compute_Bz()')
    
    def compute_Bp(self):
        return NotImplementedError('PINN::compute_Bp()')
    
    def compute_pprime(self):
        return NotImplementedError('PINN::compute_pprime()')
    
    def compute_ffprime(self):
        return NotImplementedError('PINN::compute_ffprime()')
    
    def compute_pressure_psi(self):
        return NotImplementedError('PINN::compute_pressure_psi()')
    
    def compute_Jphi_psi(self):
        return NotImplementedError('PINN::compute_Jphi_psi()')
    
    def compute_field_decay_index(self):
        return NotImplementedError('PINN::compute_field_decay_index()')
    
class NoiseLayer(nn.Module):
    def __init__(self, mean : float = 0, std : float = 1e-2):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x : torch.Tensor):
        if self.training:
            noise = torch.autograd.Variable(torch.ones_like(x).to(x.device) * self.mean + torch.randn(x.size()).to(x.device) * self.std)
            return x + noise
        else:
            return x
        
class ResBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, (in_channels + out_channels) // 2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d((in_channels + out_channels) // 2),
            nn.ReLU(),
            nn.Conv2d((in_channels + out_channels) // 2, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x : torch.Tensor):
        out = self.resblock(x)
        out = out + x
        out = self.relu(out)
        return out

class ContourRegressor(nn.Module):
    def __init__(
        self, 
        nx : int, 
        ny : int, 
        params_dim : int,
        n_PFCs :int,
        Rmin : float = 1, 
        Rmax : float = 4, 
        Zmin : float = -1.0, 
        Zmax : float = 1.0
        ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        
        self.params_dim = params_dim
        self.n_PFCs = n_PFCs
        
        self.Rmin = Rmin
        self.Zmin = Zmin
        self.Rmax = Rmax
        self.Zmax = Zmax
        
        self.encoder_params = nn.Sequential(
            nn.Linear(params_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        self.encoder_PFCs = nn.Sequential(
            nn.Linear(n_PFCs, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        self.noise = NoiseLayer(0, 0.25)
        self.conv_layer = nn.Sequential(
            nn.LayerNorm((nx,ny)),
            nn.ReLU(),
            nn.Conv2d(1,32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ResBlock(32, 32),
            nn.Conv2d(32, 64,kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ResBlock(64, 64),
        )
        
        input_dim = self.compute_hidden_dim() + 64 * 2
        
        self.cen_regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 2),
        )
        
        self.rad_regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 256),
        )
        
    def compute_hidden_dim(self):
        self.eval()
        sample_data = torch.zeros((1,self.nx,self.ny))
        sample_data = self.conv_layer(sample_data.unsqueeze(1)).view(1,-1)
        return sample_data.size()[1]
        
    def forward(self, x : torch.Tensor, x_params : torch.Tensor, x_PFCs : torch.Tensor):
        batch_size = x.size()[0]
        x = self.noise(x)
        x = self.conv_layer(x.unsqueeze(1)).view(batch_size, -1)
        
        x_params = self.noise(x_params)
        x_PFCs = self.noise(x_PFCs)
        x_params = self.encoder_params(x_params)
        x_PFCs = self.encoder_PFCs(x_PFCs)
        
        x = torch.concat([x, x_params, x_PFCs], axis = 1)
        
        cen = self.cen_regressor(x.view(batch_size, -1)).clamp(
            min = torch.Tensor([1.6,-0.1]).repeat(batch_size,1).to(x.device),
            max = torch.Tensor([1.9, 0.1]).repeat(batch_size,1).to(x.device),
        )
        
        rad = self.rad_regressor(x.view(batch_size, -1)).clamp(
            min = torch.Tensor([0.25]).unsqueeze(0).repeat(batch_size,1).to(x.device),
            max = torch.Tensor([0.9]).unsqueeze(0).repeat(batch_size,1).to(x.device),
        )
        
        cen = self.cen_regressor(x)
        rad = self.rad_regressor(x)
        
        return cen, rad
    
    def compute_rzbdys(self, x:torch.Tensor, x_params : torch.Tensor, x_PFCs : torch.Tensor, smooth : bool = True):
    
        with torch.no_grad():
            cen, rad = self.forward(x, x_params, x_PFCs)
            cen = cen.detach().squeeze(0).cpu().numpy()
            rad = rad.detach().squeeze(0).cpu().numpy()
            
            theta = np.linspace(0,2*3.142,256)
            
            # smoothing
            if smooth:
                rad_ = np.zeros((len(rad) + 4))
                rad_[2:-2] = rad
                rad_[0:2] = rad[0:2]
                rad_[-2:] = rad[-2:]
                rad = np.convolve(rad_, [0.2,0.2,0.2,0.2,0.2], 'valid')
            
            rzbdys = np.zeros((256,2))
            rzbdys[:,0] = cen[0] + rad * np.cos(theta)
            rzbdys[:,1] = cen[1] + rad * np.sin(theta)
            
        return rzbdys
    
    def compute_shape_parameters(self, x : torch.Tensor, x_params : torch.Tensor, x_PFCs : torch.Tensor, smooth : bool = False):
        with torch.no_grad():
            rzbdy = self.compute_rzbdys(x, x_params, x_PFCs, smooth)
            big_ind = 1
            small_ind = 1

            len2 = len(rzbdy)

            for i in range(len2-1):

                if (rzbdy[i,1] > rzbdy[big_ind,1]):
                    big_ind = i

                if (rzbdy[i,1] < rzbdy[small_ind,1]):
                    small_ind = i

            a = (max(rzbdy[:,0]) - min(rzbdy[:,0])) * 0.5
            R = (max(rzbdy[:,0]) + min(rzbdy[:,0])) * 0.5

            r_up = rzbdy[big_ind,0]
            r_low = rzbdy[small_ind,0]
            
            k = (rzbdy[big_ind,1]-rzbdy[small_ind,1])/a * 0.5
            triu = (R-r_up)/a
            tril = (R-r_low)/a
            
            return k, triu, tril, R, a
    
class Encoder(nn.Module):
    def __init__(self, nx : int, ny : int):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.norm = nn.Sequential(
            nn.LayerNorm((nx,ny)),
            nn.ReLU(),
        )
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, 5, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, 3),
            nn.ReLU(),
        )
        
        input_dim = self.compute_hidden_dim()
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64,1),
        )
    
    def compute_hidden_dim(self):
        sample_data = torch.zeros((1,self.nx,self.ny))
        sample_data = self.norm(sample_data)
        sample_data = self.conv_layer(sample_data.unsqueeze(1)).view(1,-1)
        return sample_data.size()[1]
        
    def forward(self, x : torch.Tensor):
        x = self.norm(x)
        x = self.conv_layer(x.unsqueeze(1))
        x = self.regressor(x.view(x.size()[0], -1))
        return x
    
class Normalization(nn.Module):
    def __init__(self, nx : int, ny : int, limiter_mask : torch.Tensor):
        super().__init__()
        self.nx = nx
        self.ny = ny
        
        self.axis_regressor = Encoder(nx,ny)
        self.bndry_regressor = Encoder(nx,ny)
        
        self.limiter_mask = limiter_mask
        
    def forward(self, x : torch.Tensor):
        
        batch_size = x.size()[0]
        
        # min and max value : psi_a and psi_b should be bounded
        x_min = x.view(batch_size,-1).min(1)[0].view(batch_size, 1)
        x_max = x.view(batch_size,-1).max(1)[0].view(batch_size, 1)
        
        # masking
        x = x * self.limiter_mask.to(x.device)
        
        x_a = self.axis_regressor(x).clamp(min = x_min, max = torch.Tensor([0]).unsqueeze(0).repeat(batch_size, 1).to(x.device))
        x_b = self.bndry_regressor(x).clamp(min = x_min, max = x_max)
        
        x = torch.clamp((x - x_a.unsqueeze(-1)) / (x_b.unsqueeze(-1) - x_a.unsqueeze(-1)), min = 0, max = 1.5)
        return x
    
    def predict_critical_value(self, x : torch.Tensor):
        with torch.no_grad():
            x *= self.limiter_mask.to(x.device)
            
            batch_size = x.size()[0]
            x_min = x.view(batch_size,-1).min(1)[0].view(batch_size, 1)
            x_max = x.view(batch_size,-1).max(1)[0].view(batch_size, 1)
            
            x_a = self.axis_regressor(x).clamp(min = x_min, max = torch.Tensor([0]).unsqueeze(0).repeat(batch_size, 1).to(x.device))
            x_b = self.bndry_regressor(x).clamp(min = x_min, max = x_max)
            return x_a, x_b
    
class PINN(AbstractPINN):
    def __init__(
        self, 
        R : np.ndarray,
        Z : np.ndarray,
        Rc : float,
        params_dim : int,
        n_PFCs :int,
        hidden_dim : int, 
        alpha_m : float = 2.0, 
        alpha_n : float = 1.0,
        beta_m : float = 2.0,
        beta_n : float = 1.0,
        lamda : float = 1.0,
        beta : float = 0.5,
        nx : int = 65,
        ny : int = 65,
        ):
        super().__init__()
        
        # Re-scale
        self.Jphi_scale = 1 / Rc ** 2
        self.psi_scale = 4 * math.pi * 10 **(-7) * 1 * Rc * (10 ** 6)
        
        # input data setting
        self.params_dim = params_dim
        self.n_PFCs = n_PFCs
        
        # position info
        self.R1D = R[0,:]
        self.Z1D = Z[:,0]
        self.R2D = R
        self.Z2D = Z
        self.dr = R[0,1] - R[0,0]
        self.dz = Z[1,0] - Z[0,0]
        
        # position vectors for computation
        self.r = torch.from_numpy(R).float().unsqueeze(0)
        self.r.requires_grad = True
        
        self.z = torch.from_numpy(Z).float().unsqueeze(0)
        self.z.requires_grad = True
        
        # Fitting current profile
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.beta_m = beta_m
        self.beta_n = beta_n
        
        # adjusted parameters for current profile
        self.lamda = nn.Parameter(torch.Tensor([lamda]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=True)
        
        self.Rc = Rc
        
        # output dimension
        self.nx = nx
        self.ny = ny
        
        # constraint : limiter
        limiter_mask = compute_KSTAR_limiter_mask(R, Z)
        self.limiter_mask = torch.from_numpy(limiter_mask).unsqueeze(0)
        
        # Encoder : embedd input data to the latent vector space
        self.encoder_pos = nn.Sequential(
            nn.Linear(2 * nx * ny, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
        )
        
        self.encoder_params = nn.Sequential(
            nn.Linear(params_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
        )
        
        self.encoder_PFCs = nn.Sequential(
            nn.Linear(n_PFCs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
        )
        self.connector = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
        )
        
        # Decoder : predict the psi from the encoded vectors
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, nx * ny),
        )
        
        # normalization
        self.norms = Normalization(nx,ny, self.limiter_mask)
        
        # boundary loss
        self.bndry_indices = np.concatenate(
            [
                [(x,0) for x in range(nx)],
                [(x,ny-1) for x in range(nx)],
                [(0,y) for y in range(ny)],
                [(nx-1,y) for y in range(ny)]
            ]
        )
        
        self.PF_coils = PF_coils
        
    def forward(self, x_params : torch.Tensor, x_PFCs : torch.Tensor):
        
        self.r = self.r.to(x_params.device)
        self.z = self.z.to(x_params.device)
        
        # position info
        x_r = self.r.repeat(x_params.size()[0],1,1).view(x_params.size()[0],-1)
        x_z = self.z.repeat(x_params.size()[0],1,1).view(x_params.size()[0],-1)
        
        x_pos = torch.concat([x_r, x_z], dim = 1)
        x_pos = self.encoder_pos(x_pos)
        
        # betap, q95, li info
        x_params = self.encoder_params(x_params)
        
        # PFC coil current
        x_PFCs = self.encoder_PFCs(x_PFCs)
        
        x = torch.concat((x_pos, x_params, x_PFCs), dim = 1)
        x = self.connector(x)
        x = self.decoder(x).view(x.size()[0], self.nx, self.ny)
        
        # to avoid nan or inf
        x = torch.clamp(x, -10.0, 10.0)
        x = torch.nan_to_num(x, nan = 0, posinf = 10.0, neginf = -10.0)
        return x
    
    def predict(self, x_params : torch.Tensor, x_PFCs : torch.Tensor):
        with torch.no_grad():
            x = self.forward(x_params, x_PFCs)
        return x
    
    def compute_boundary_loss(self, psi : torch.Tensor, x_PFCs : torch.Tensor):
        
        bc_loss = 0
        
        for x,y in self.bndry_indices:
            psi_b = psi[:,x,y]
            psi_b_pred = 0
            
            # boundary magnetic flux induced by PFC coil current
            for coil_idx, coil_comp in enumerate(self.PF_coils.keys()):
                Ic = x_PFCs[:,coil_idx]
                x_coil, y_coil = self.PF_coils[coil_comp]
                g = compute_Green_function(x_coil, y_coil, self.R2D[x,y].item(), self.Z2D[x,y].item())
                psi_b_pred += Ic * g

            # boundary magnetic flux induced by plasma current
            Jphi = self.compute_Jphi_GS(psi) * self.Ip_scale
            g_matrix = compute_Green_function(self.R2D.ravel(), self.Z2D.ravel(), self.R2D[x,y].item(), self.Z2D[x,y].item())
            g_matrix = torch.as_tensor(g_matrix, dtype = torch.float).view(self.R2D.shape).to(Jphi.device)
            g_matrix = torch.nan_to_num(g_matrix, nan = 0, posinf = 0, neginf = 0)
            psi_p = torch.sum(Jphi * g_matrix, dim = (1,2)) * self.dr * self.dz
            psi_b_pred += psi_p
            
            bc_loss += (psi_b_pred - psi_b) ** 2
        
        bc_loss /= len(self.bndry_indices)
        bc_loss = torch.sqrt(bc_loss).sum()
        return bc_loss

    def compute_Jphi_GS(self, psi : torch.Tensor):
        Jphi = eliptic_operator(psi, self.r, self.z) * (-1)
        Jphi /= self.r
        return Jphi * self.compute_plasma_region(psi) 
        
    def compute_Jphi(self, psi : torch.Tensor):
        return compute_Jphi(self.norms(psi), self.r / self.Rc, self.alpha_m, self.alpha_n, self.beta_m, self.beta_n, self.lamda, self.beta) * self.compute_plasma_region(psi)
        
    def compute_plasma_region(self, psi:torch.Tensor):
        return compute_plasma_region(self.norms(psi)) * self.limiter_mask.to(psi.device)
    
    def compute_GS_loss(self, psi : torch.Tensor):
        Jphi = compute_Jphi(self.norms(psi), self.r / self.Rc, self.alpha_m, self.alpha_n, self.beta_m, self.beta_n, self.lamda.detach(), self.beta.detach()) * self.compute_plasma_region(psi)
        return compute_grad_shafranov_loss(psi, self.r, self.z, Jphi, self.Rc, self.psi_scale)
    
    def compute_plasma_current(self, psi:torch.Tensor):
        Jphi = compute_Jphi(self.norms(psi).detach(), self.r / self.Rc, self.alpha_m, self.alpha_n, self.beta_m, self.beta_n, self.lamda, self.beta) * self.compute_plasma_region(psi)
        Ip = torch.sum(Jphi, dim = (1,2)).view(-1,1) * self.dr * self.dz * self.Jphi_scale
        return Ip
    
    def compute_constraint_loss(self, psi : torch.Tensor, Ip : torch.Tensor, betap : torch.Tensor):
        Ip_constraint = self.compute_constraint_loss_Ip(psi, Ip)
        betap_constraint = self.compute_constraint_loss_betap(psi, Ip, betap)
        return Ip_constraint + betap_constraint
    
    def compute_constraint_loss_Ip(self, psi : torch.Tensor, Ip : float):
        Ip_compute = self.compute_plasma_current(psi)
        return torch.norm(Ip_compute- Ip)
    
    def compute_constraint_loss_betap(self, psi : torch.Tensor, Ip : torch.Tensor, betap : torch.Tensor):
        scale = 8 * math.pi / self.Rc ** 2 / Ip ** 2
        pressure = compute_p_psi(psi * self.compute_plasma_region(psi), self.r / self.Rc, self.alpha_m, self.alpha_n, self.beta_m, self.beta_n, self.lamda, self.beta)
        betap_compute = torch.sum(pressure, dim = (1,2)).view(-1,1) * self.dr * self.dz * scale
        return torch.norm(betap_compute - betap)    
    
    def compute_betap(self, psi : torch.Tensor, Ip : torch.Tensor):
        scale = 8 * math.pi / self.Rc ** 2 / Ip.abs() ** 2
        pressure = compute_p_psi(self.norms(psi).detach(), self.r / self.Rc, self.alpha_m, self.alpha_n, self.beta_m, self.beta_n, self.lamda, self.beta)
        pressure = pressure * self.compute_plasma_region(psi).detach()
        return torch.sum(pressure, dim = (1,2)).view(-1,1) * self.dr * self.dz * scale
    
    def compute_Br(self, x_params : torch.Tensor, x_PFCs : torch.Tensor):
        psi = self.forward(x_params, x_PFCs)
        Br = (-1) * gradient(psi, self.z) / self.r
        return Br
    
    def compute_Bz(self, x_params : torch.Tensor, x_PFCs : torch.Tensor):
        psi = self.forward(x_params, x_PFCs)
        Bz = gradient(psi, self.r) / self.r
        return Bz
    
    def compute_Bp(self, x_params : torch.Tensor, x_PFCs : torch.Tensor):
        Br = self.compute_Br(x_params, x_PFCs)
        Bz = self.compute_Bz(x_params, x_PFCs)
        Bp = torch.sqrt(Br ** 2 + Bz ** 2)
        return Bp
    
    def compute_Btor(self, x_params : torch.Tensor, x_PFCs : torch.Tensor):
        psi = self.forward(x_params, x_PFCs)
        psi_norm = self.norms(psi)
        
        psi_np = psi.detach().squeeze(0).cpu().numpy()
        psi_norm_np = psi_norm.detach().squeeze(0).cpu().numpy()
        
        _compute_ffprime = lambda x : compute_ffprime(x, self.R1D / self.Rc, self.alpha_m, self.alpha_n, self.beta_m, self.beta_n, self.lamda.detach().cpu().item(), self.beta.detach().cpu().item())

        psi_norm_np_1d = psi_norm_np.reshape(-1,)
        psi_np_1d = psi_np.reshape(-1,)
        
        fpol = np.zeros_like(psi_np).reshape(-1,)
        
        for idx, (psi_n, psi) in enumerate(zip(psi_norm_np_1d, psi_np_1d)):
            val, _ = quad(_compute_ffprime, psi_n, 1.0)
            val *= psi / psi_n
            
            fpol[idx] = val
            
        fpol = np.reshape(fpol, psi_np.shape)
        Btol = fpol / self.R2D
        return Btol
    
    def compute_pprime(self):
        psi = np.linspace(0,1,64)
        return psi, compute_pprime(psi, self.R1D / self.Rc, self.alpha_m, self.alpha_n, self.beta_m, self.beta_n, self.lamda.detach().cpu().item(), self.beta.detach().cpu().item())

    def compute_ffprime(self):
        psi = np.linspace(0,1,64)
        return psi, compute_ffprime(psi, self.R1D / self.Rc, self.alpha_m, self.alpha_n, self.beta_m, self.beta_n, self.lamda.detach().cpu().item(), self.beta.detach().cpu().item())
    
    def compute_fpol(self, psi : torch.Tensor):
        device = psi.device
        psi_norm = self.norms(psi) * self.limiter_mask.to(psi.device)
        
        psi_np = psi.detach().squeeze(0).cpu().numpy()
        psi_norm_np = psi_norm.detach().squeeze(0).cpu().numpy()
        
        mu = 4 * math.pi * 10 **(-7)
        
        _compute_ffprime = lambda x : self.Jphi_scale * mu * self.Rc * compute_ffprime(x, self.R1D / self.Rc, self.alpha_m, self.alpha_n, self.beta_m, self.beta_n, self.lamda.detach().cpu().item(), self.beta.detach().cpu().item())

        psi_norm_np_1d = psi_norm_np.reshape(-1,)
        psi_np_1d = psi_np.reshape(-1,)
        
        fpol = np.zeros_like(psi_np).reshape(-1,)
        
        x_a, x_b = self.norms.predict_critical_value(psi)
        
        for idx, (psi_n, psi) in enumerate(zip(psi_norm_np_1d, psi_np_1d)):
            val, _ = quad(_compute_ffprime, 0, psi_n)
            val *= (x_b - x_a) * (-1)
            fpol[idx] = val
            
        fpol = np.reshape(fpol, psi_np.shape)
        fpol = torch.from_numpy(fpol).unsqueeze(0).to(device)
        return fpol
    
    def compute_pressure_psi(self):
        psi = np.linspace(0,1,64)
        pressure = np.zeros_like(psi)
        _compute_pprime = lambda x : compute_pprime(x, self.R1D / self.Rc, self.alpha_m, self.alpha_n, self.beta_m, self.beta_n, self.lamda.detach().cpu().item(), self.beta.detach().cpu().item())
        
        for idx in range(1, len(psi)-1):
            val, _ = quad(_compute_pprime, 0.0, psi[idx])
            pressure[idx] = val
            
        return psi, pressure
    
    def compute_Jphi_psi(self, psi : torch.Tensor):
        Jphi = self.compute_Jphi(psi)
        psi = self.norms(psi)
        
        if psi.ndim == 3:
            psi = psi.squeeze(0)
            
        psi = psi.detach().cpu().numpy()
        psi_norm = np.linspace(0,1,32)
        Jphi_norm = []
        Jphi = Jphi.squeeze(0).detach().cpu().numpy()
        
        for p in psi_norm:
            contours = measure.find_contours(psi, p)
        
            for contour in contours:
                Jphi_list = []
                for idx in range(len(contour)):
                    Jphi_list.append(Jphi[int(contour[idx,1].item()), int(contour[idx,0].item())])
            
            if len(contours) > 0:
                Jphi_avg = np.mean(Jphi_list)
                Jphi_norm.append(Jphi_avg)   
            else:
                Jphi_avg = 0
                Jphi_norm.append(Jphi_avg)
                
        # interpolation
        interp_fn = interp1d(psi_norm, Jphi_norm, kind = 'cubic', fill_value = 'extrapolate')
        psi = np.linspace(0, 1, 64)
        Jphi = interp_fn(psi)
        
        return psi, Jphi
    
    def compute_q_psi(self, psi : torch.Tensor):
        fpol = self.compute_fpol(psi)
        
        Br = (-1) * gradient(psi, self.z) / self.r
        Bz = gradient(psi, self.r) / self.r
        
        Bp = Br ** 2 + Bz ** 2
        Bp = torch.sqrt(Bp)
        
        (r_axis, z_axis), _ = self.find_axis(psi, 1e-4)
        r_axis /= self.Rc
        z_axis /= self.Rc
        
        dl = (self.r - r_axis) ** 2 + (self.z - z_axis) ** 2
        dl = torch.sqrt(dl)
        
        q = fpol * dl / self.r ** 2 / Bp
        q = q.detach().cpu().squeeze(0).numpy()
        q_psi = []
        
        psi = self.norms(psi)
        
        if psi.ndim == 3:
            psi = psi.squeeze(0)
        
        psi = psi.detach().cpu().numpy()
        psi_norm = np.linspace(0,1,32)
        
        for p in psi_norm:
            contours = measure.find_contours(psi, p)
            for contour in contours:
                q_list = []
                for idx in range(len(contour)):
                    q_list.append(q[int(contour[idx,1].item()), int(contour[idx,0].item())])
                    
            
            if len(contours) > 0:
                q_avg = np.mean(q_list)
                q_psi.append(q_avg)   
            else:
                q_avg = 0
                q_psi.append(q_avg)
            
        # interpolation
        interp_fn = interp1d(psi_norm, q_psi, kind = 'cubic', fill_value = 'extrapolate')
        psi = np.linspace(0, 1, 64)
        q_psi_1d = interp_fn(psi)
        
        return psi, q_psi_1d
    
    def compute_field_decay_index(self, x_params : torch.Tensor, x_PFCs : torch.Tensor):
        Br = self.compute_Br(x_params, x_PFCs)
        Brz = gradient(Br, self.z)
        
        Bz = self.compute_Bz(x_params, x_PFCs)
        Bzr = gradient(Bz, self.r)
    
        n_decay = (-1) * self.r * Bzr / Bz + (-1) * self.r * Brz / Bz
        n_decay *= 0.5
        return n_decay
    
    def find_axis(self, psi : torch.Tensor, eps : float = 1e-4):
        det = compute_det(psi, self.r, self.z)
        grad = compute_grad2(psi, self.r, self.z)
        
        mask_grad = grad.le(eps)
        mask_det = det.ge(0)
        
        limiter_mask = compute_KSTAR_limiter_mask(self.R2D, self.Z2D, 0)
        limiter_mask = torch.from_numpy(limiter_mask).unsqueeze(0)
        
        psi_masked = mask_det * mask_grad * psi * limiter_mask.to(psi.device)
        psi_axis = psi_masked.min()
        
        r_axis = self.R2D.ravel()[torch.argmin(psi_masked).item()]
        z_axis = self.Z2D.ravel()[torch.argmin(psi_masked).item()]
                
        return (r_axis, z_axis), psi_axis
        
    def find_xpoints(self, psi : torch.Tensor, eps : float = 1e-4):
        det = compute_det(psi, self.r, self.z)
        grad = compute_grad2(psi, self.r, self.z)
        
        mask_grad = grad.le(eps)
        mask_det = det.le(0)
        
        limiter_mask = compute_KSTAR_limiter_mask(self.R2D, self.Z2D, 0)
        limiter_mask = torch.from_numpy(limiter_mask).unsqueeze(0)
        
        psi_masked = mask_det * mask_grad * psi * limiter_mask.to(psi.device)
        indices_xpts = torch.argwhere(psi_masked.flatten() > 0.1)
        
        if len(indices_xpts) == 0:
            return []
    
        xpts = []
        for idx in indices_xpts.detach().cpu().numpy():
            psi_xpt = psi_masked.flatten()[idx].detach().cpu().item()
            r_xpt = self.R2D.ravel()[idx]
            z_xpt = self.Z2D.ravel()[idx]
            xpts.append((r_xpt, z_xpt, psi_xpt))
        
        xpts = self.remove_dup(xpts, 1e-3)
        
        return xpts
    
    def remove_dup(self, points, eps : float = 1e-5):
        
        result = []
        
        for n, p in enumerate(points):
            dup = False
            for p2 in result:
                if math.sqrt((p[0] - p2[0]) ** 2 + (p[1] - p2[1]) ** 2) < eps:
                    dup = True  # Duplicate
                    break
            if not dup:
                result.append(p)  # Add to the list
        return result