'''
    Reference
    - code : https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    - Question : single-step (only use linear layer) vs multi-step (use Transformer Decoder Layer)
    - Wu, N., Green, B., Ben, X., O'banion, S. (2020). 'Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case'
'''
import torch, math
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional, Dict
from pytorch_model_summary import summary

# Transformer model
class NoiseLayer(nn.Module):
    def __init__(self, mean : float = 0, std : float = 1e-2):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x : torch.Tensor):
        if self.training:
            noise = Variable(torch.ones_like(x).to(x.device) * self.mean + torch.randn(x.size()).to(x.device) * self.std)
            return x + noise
        else:
            return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, max_len : int = 5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1) # (max_len, 1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() # (d_model // 2, )

        pe[:,0::2] = torch.sin(position * div_term)

        if d_model % 2 != 0:
            pe[:,1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0,1) # shape : (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor):
        # x : (seq_len, batch_size, n_features)
        return x + self.pe[:x.size(0), :, :]

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class CNNencoder(nn.Module):
    def __init__(self, input_dim : int, feature_dim : int, kernel_size : int, padding : int, reduction : bool):
        super().__init__()
        dk = 1 if reduction else 0
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=feature_dim, kernel_size= kernel_size + dk, stride = 1, padding = padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim,out_channels=feature_dim, kernel_size= kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
    def forward(self, x : torch.Tensor):
        return self.encoder(x)
'''
class Transformer(nn.Module):
    def __init__(
        self, 
        n_layers : int = 2, 
        n_heads : int = 8, 
        dim_feedforward : int = 1024, 
        dropout : float = 0.1,        
        RIN : bool = False,
        input_0D_dim : int = 12,
        input_0D_seq_len : int = 20,
        input_ctrl_dim : int = 14,
        input_ctrl_seq_len : int = 24,
        output_0D_pred_len : int = 4,
        output_0D_dim : int = 12,
        feature_0D_dim : int = 128,
        feature_ctrl_dim : int = 128,
        range_info : Optional[Dict] = None,
        noise_mean : float = 0,
        noise_std : float = 0.81,
        kernel_size : int = 3,
        ):
        
        super(Transformer, self).__init__()
        
        # input information
        self.input_0D_dim = input_0D_dim
        self.input_0D_seq_len = input_0D_seq_len
        self.input_ctrl_dim = input_ctrl_dim
        self.input_ctrl_seq_len = input_ctrl_seq_len
        
        # output information
        self.output_0D_pred_len = output_0D_pred_len
        self.output_0D_dim = output_0D_dim
        
        # source mask
        self.src_mask_0D = None
        self.src_mask_ctrl = None
        
        # transformer info
        self.feature_0D_dim = feature_0D_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.feature_ctrl_dim = feature_ctrl_dim
        
        self.RIN = RIN
        
        self.noise = NoiseLayer(mean = noise_mean, std = noise_std)
        
        if kernel_size // 2 == 0:
            print("kernel sholud be odd number")
            kernel_size += 1
        padding = (kernel_size - 1) // 2
        
        # 0D data encoder
        self.encoder_input_0D = nn.Sequential(
            nn.Conv1d(in_channels=input_0D_dim, out_channels=feature_0D_dim // 2, kernel_size= kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_0D_dim//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_0D_dim // 2, out_channels=feature_0D_dim, kernel_size= kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_0D_dim),
            nn.ReLU()
        )
        
        self.pos_0D = PositionalEncoding(d_model = feature_0D_dim, max_len = input_0D_seq_len)
        
        self.enc_0D = nn.TransformerEncoderLayer(
            d_model = feature_0D_dim, 
            nhead = n_heads, 
            dropout = dropout,
            dim_feedforward = dim_feedforward,
            activation = GELU()
        )
        
        self.trans_enc_0D = nn.TransformerEncoder(self.enc_0D, num_layers=n_layers)
        
        # ctrl data encoder
        self.encoder_input_ctrl = nn.Sequential(
            nn.Conv1d(in_channels=input_ctrl_dim, out_channels=feature_ctrl_dim // 2, kernel_size= kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_ctrl_dim//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_ctrl_dim // 2, out_channels=feature_ctrl_dim, kernel_size= kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_ctrl_dim),
            nn.ReLU()
        )
        
        self.pos_ctrl = PositionalEncoding(d_model = feature_ctrl_dim, max_len = input_ctrl_seq_len)
        self.enc_ctrl = nn.TransformerEncoderLayer(
            d_model = feature_ctrl_dim, 
            nhead = n_heads, 
            dropout = dropout,
            dim_feedforward = dim_feedforward,
            activation = GELU()
        )
        
        self.trans_enc_ctrl = nn.TransformerEncoder(self.enc_ctrl, num_layers=n_layers)

        # FC decoder
        # sequence length reduction
        self.lc_0D_seq = nn.Linear(input_0D_seq_len, output_0D_pred_len)
        self.lc_ctrl_seq = nn.Linear(input_ctrl_seq_len, output_0D_pred_len)
        
        # dimension reduction
        self.lc_feat= nn.Linear(feature_0D_dim + feature_ctrl_dim, output_0D_dim)
        
        # Reversible Instance Normalization
        if self.RIN:
            self.affine_weight_0D = nn.Parameter(torch.ones(1, 1, input_0D_dim))
            self.affine_bias_0D = nn.Parameter(torch.zeros(1, 1, input_0D_dim))
            
        self.range_info = range_info
        
        if range_info:
            self.range_min = torch.Tensor([range_info[key][0] * 0.1 for key in range_info.keys()])
            self.range_max = torch.Tensor([range_info[key][1] * 10.0 for key in range_info.keys()])
        else:
            self.range_min = None
            self.range_max = None
            
        # initialize
        self.init_weights()
    
    def init_weights(self):
        
        initrange = 0.1    
        self.lc_feat.bias.data.zero_()
        self.lc_feat.weight.data.uniform_(-initrange, initrange)
        
        self.lc_0D_seq.bias.data.zero_()
        self.lc_0D_seq.weight.data.uniform_(-initrange, initrange)
        
        self.lc_ctrl_seq.bias.data.zero_()
        self.lc_ctrl_seq.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x_0D : torch.Tensor, x_ctrl : torch.Tensor):
        
        b = x_0D.size()[0]
        
        # add noise to robust performance
        x_0D = self.noise(x_0D)
        
        if self.RIN:
            means_0D = x_0D.mean(1, keepdim=True).detach()
            x_0D = x_0D - means_0D
            stdev_0D = torch.sqrt(torch.var(x_0D, dim=1, keepdim=True, unbiased=False) + 1e-3).detach()
            x_0D /= stdev_0D
            x_0D = x_0D * self.affine_weight_0D + self.affine_bias_0D
            
        # path : 0D data
        # encoding : (N, T, F) -> (N, T, d_model)
        x_0D = self.encoder_input_0D(x_0D.permute(0,2,1)).permute(0,2,1)
        
        # (T, N, d_model)
        x_0D = x_0D.permute(1,0,2)
        self.src_mask_0D = self._generate_square_subsequent_mask(len(x_0D)).to(x_0D.device)
            
        # positional encoding for time axis : (T, N, d_model)
        x_0D = self.pos_0D(x_0D)
        
        # transformer encoding layer : (T, N, d_model)
        x_0D = self.trans_enc_0D(x_0D, self.src_mask_0D.to(x_0D.device))
        
        # (N, T, d_model)
        x_0D = x_0D.permute(1,0,2)
        
        # path : ctrl data
        x_ctrl = self.encoder_input_ctrl(x_ctrl.permute(0,2,1)).permute(0,2,1)
        
        # (T, N, d_model)
        x_ctrl = x_ctrl.permute(1,0,2)
        self.src_mask_ctrl = self._generate_square_subsequent_mask(len(x_ctrl)).to(x_ctrl.device)
            
        # positional encoding for time axis : (T, N, d_model)
        x_ctrl = self.pos_ctrl(x_ctrl)
        
        # transformer encoding layer : (T, N, d_model)
        x_ctrl = self.trans_enc_ctrl(x_ctrl, self.src_mask_ctrl.to(x_ctrl.device))
        
        # (N, T, d_model)
        x_ctrl = x_ctrl.permute(1,0,2)
        
        # (N, d_model, T_)
        x_0D = self.lc_0D_seq(x_0D.permute(0,2,1))
        
        # (N, T_, d_model)
        x_0D = x_0D.permute(0,2,1)
        
        # (N, d_model, T_)
        x_ctrl = self.lc_ctrl_seq(x_ctrl.permute(0,2,1))
        
        # (N, T_, d_model)
        x_ctrl = x_ctrl.permute(0,2,1)
        
        x = torch.concat([x_0D, x_ctrl], axis = 2)
        
        # dim reduction
        x = self.lc_feat(x)
    
        # RevIN for considering data distribution shift
        if self.RIN:
            x = x - self.affine_bias_0D
            x = x / (self.affine_weight_0D + 1e-6)
            x = x * stdev_0D
            x = x + means_0D  
        
        # clamping : output range
        # x = torch.clamp(x, min = self.range_min.to(x.device), max = self.range_max.to(x.device))
        x = torch.clamp(x, min = -10.0, max = 10.0)
        
        # remove nan value for stability
        x = torch.nan_to_num(x, nan = 0)
        
        return x

    def _generate_square_subsequent_mask(self, size : int):
        mask = (torch.triu(torch.ones(size,size))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def summary(self):
        sample_0D = torch.zeros((1, self.input_0D_seq_len, self.input_0D_dim))
        sample_ctrl = torch.zeros((1, self.input_ctrl_seq_len, self.input_ctrl_dim))
        summary(self, sample_0D, sample_ctrl, batch_size = 1, show_input = True, print_summary=True)
'''     
class Transformer(nn.Module):
    def __init__(
        self, 
        n_layers : int = 2, 
        n_heads : int = 8, 
        dim_feedforward : int = 1024, 
        dropout : float = 0.1,        
        RIN : bool = False,
        input_0D_dim : int = 12,
        input_0D_seq_len : int = 20,
        input_ctrl_dim : int = 14,
        input_ctrl_seq_len : int = 24,
        output_0D_pred_len : int = 4,
        output_0D_dim : int = 12,
        feature_dim : int = 128,
        range_info : Optional[Dict] = None,
        noise_mean : float = 0,
        noise_std : float = 0.81,
        kernel_size : int = 3,
        ):
        
        super(Transformer, self).__init__()
        
        # input information
        self.input_0D_dim = input_0D_dim
        self.input_0D_seq_len = input_0D_seq_len
        self.input_ctrl_dim = input_ctrl_dim
        self.input_ctrl_seq_len = input_ctrl_seq_len
        
        # output information
        self.output_0D_pred_len = output_0D_pred_len
        self.output_0D_dim = output_0D_dim
        
        # source mask
        self.src_mask = None
        self.src_mask = None
        
        # transformer info
        self.feature_dim = feature_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        self.RIN = RIN
        
        self.noise = NoiseLayer(mean = noise_mean, std = noise_std)
        
        self.input_seq_len = min(input_0D_seq_len, input_ctrl_seq_len)
        
        if kernel_size // 2 == 0:
            print("kernel sholud be odd number")
            kernel_size += 1
            
        padding = (kernel_size - 1) // 2
        
        # Convolution layer for extracting the temporal componenets
        self.encoder_input_0D = CNNencoder(input_0D_dim, feature_dim//2, kernel_size, padding, False)
        self.encoder_input_ctrl = CNNencoder(input_ctrl_dim, feature_dim//2, kernel_size, padding, True if input_ctrl_seq_len > input_0D_seq_len else False)
        
        self.pos = PositionalEncoding(d_model = feature_dim, max_len = self.input_seq_len)
        self.enc = nn.TransformerEncoderLayer(
            d_model = feature_dim, 
            nhead = n_heads, 
            dropout = dropout,
            dim_feedforward = dim_feedforward,
            activation = GELU()
        )
        
        self.trans_enc = nn.TransformerEncoder(self.enc, num_layers=n_layers)
        
        # FC decoder
        # sequence length reduction
        self.lc_seq = nn.Linear(self.input_seq_len, output_0D_pred_len)
        
        # dimension reduction
        self.lc_feat= nn.Linear(feature_dim, output_0D_dim)
        
        # Reversible Instance Normalization
        if self.RIN:
            self.affine_weight_0D = nn.Parameter(torch.ones(1, 1, input_0D_dim))
            self.affine_bias_0D = nn.Parameter(torch.zeros(1, 1, input_0D_dim))
            
        self.range_info = range_info
        
        if range_info:
            self.range_min = torch.Tensor([range_info[key][0] * 0.1 for key in range_info.keys()])
            self.range_max = torch.Tensor([range_info[key][1] * 10.0 for key in range_info.keys()])
        else:
            self.range_min = None
            self.range_max = None
            
        # initialize
        self.init_weights()
        
    def init_weights(self):
        
        initrange = 0.1    
        self.lc_feat.bias.data.zero_()
        self.lc_feat.weight.data.uniform_(-initrange, initrange)
        
        self.lc_seq.bias.data.zero_()
        self.lc_seq.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x_0D : torch.Tensor, x_ctrl : torch.Tensor):
            
        # add noise to robust performance
        x_0D = self.noise(x_0D)
        
        if self.RIN:
            means_0D = x_0D.mean(1, keepdim=True).detach()
            x_0D = x_0D - means_0D
            stdev_0D = torch.sqrt(torch.var(x_0D, dim=1, keepdim=True, unbiased=False) + 1e-3).detach()
            x_0D /= stdev_0D
            x_0D = x_0D * self.affine_weight_0D + self.affine_bias_0D
            
        # path : 0D data
        # encoding : (N, T, F) -> (N, T, d_model)
        x_0D = self.encoder_input_0D(x_0D.permute(0,2,1)).permute(0,2,1)
        x_ctrl = self.encoder_input_ctrl(x_ctrl.permute(0,2,1)).permute(0,2,1)
        
        x = torch.concat([x_0D, x_ctrl], dim = 2)
        
        # (T, N, d_model)
        x = x.permute(1,0,2)
    
        self.src_mask = self._generate_square_subsequent_mask(len(x)).to(x.device)
            
        # positional encoding for time axis : (T, N, d_model)
        x = self.pos(x)
        
        # transformer encoding layer : (T, N, d_model)
        x = self.trans_enc(x, self.src_mask.to(x.device))
        
        # (N, T, d_model)
        x = x.permute(1,0,2)
    
        # (N, d_model, T_)
        x = self.lc_seq(x.permute(0,2,1))
        x = torch.nn.functional.relu(x)
        
        # (N, T_, d_model)
        x = x.permute(0,2,1)
        
        # dim reduction
        x = self.lc_feat(x)
    
        # RevIN for considering data distribution shift
        if self.RIN:
            x = x - self.affine_bias_0D
            x = x / (self.affine_weight_0D + 1e-6)
            x = x * stdev_0D
            x = x + means_0D  
        
        # clamping : output range
        # x = torch.clamp(x, min = self.range_min.to(x.device), max = self.range_max.to(x.device))
        x = torch.clamp(x, min = -10.0, max = 10.0)
        
        # remove nan value for stability
        x = torch.nan_to_num(x, nan = 0)
        
        return x

    def _generate_square_subsequent_mask(self, size : int):
        mask = (torch.triu(torch.ones(size,size))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def summary(self):
        sample_0D = torch.zeros((1, self.input_0D_seq_len, self.input_0D_dim))
        sample_ctrl = torch.zeros((1, self.input_ctrl_seq_len, self.input_ctrl_dim))
        summary(self, sample_0D, sample_ctrl, batch_size = 1, show_input = True, print_summary=True)