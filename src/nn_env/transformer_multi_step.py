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

class Transformer(nn.Module):
    def __init__(
        self, 
        n_layers : int = 2, 
        n_heads : int = 8, 
        dim_feedforward : int = 1024, 
        dropout : float = 0.1,        
        RIN : bool = False,
        input_0D_dim : int = 12,
        input_ctrl_dim : int = 14,
        input_seq_len : int = 16,
        output_pred_len : int = 4,
        output_0D_dim : int = 12,
        feature_dim : int = 128,
        range_info : Optional[Dict] = None,
        noise_mean : float = 0,
        noise_std : float = 0.81,
        ):
        
        super(Transformer, self).__init__()
        
        # input information
        self.input_0D_dim = input_0D_dim
        self.input_seq_len = input_seq_len
        self.input_ctrl_dim = input_ctrl_dim
        
        # output information
        self.output_pred_len = output_pred_len
        self.output_0D_dim = output_0D_dim
        
        # source mask
        self.src_mask = None
        self.src_dec_mask = None
        self.tgt_dec_mask = None
        
        # transformer info
        self.feature_dim = feature_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.RIN = RIN
        
        self.noise = NoiseLayer(mean = noise_mean, std = noise_std)
        
        # 0D data encoder
        self.encoder_input = nn.Sequential(
            nn.Linear(in_features=input_0D_dim + input_ctrl_dim, out_features=feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim //2, feature_dim)
        )
        
        # positional embedding
        self.pos = PositionalEncoding(d_model = feature_dim, max_len = input_seq_len)
        
        # Encoder layer
        self.enc = nn.TransformerEncoderLayer(
            d_model = feature_dim, 
            nhead = n_heads, 
            dropout = dropout,
            dim_feedforward = dim_feedforward,
            activation = GELU()
        )
        
        self.trans_enc = nn.TransformerEncoder(self.enc, num_layers=n_layers)
        
        # Decoder
        self.decoder_input = nn.Sequential(
            nn.Linear(in_features=input_0D_dim + input_ctrl_dim, out_features=feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim //2, feature_dim)
        )
        
        self.dec = nn.TransformerDecoderLayer(
            d_model = feature_dim, 
            nhead = n_heads,
            dropout = dropout,
            dim_feedforward = dim_feedforward,
            activation = GELU()
        )
        
        self.trans_dec = nn.TransformerDecoder(decoder_layer = self.dec, num_layers = n_layers)
        
        # dimension reduction for feature dimension
        self.lc_feat= nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, output_0D_dim),
        )
        
        # Reversible Instance Normalization
        if self.RIN:
            self.affine_weight_0D = nn.Parameter(torch.ones(1, 1, input_0D_dim))
            self.affine_bias_0D = nn.Parameter(torch.zeros(1, 1, input_0D_dim))
            
            self.affine_weight_ctrl = nn.Parameter(torch.ones(1, 1, input_ctrl_dim))
            self.affine_bias_ctrl = nn.Parameter(torch.zeros(1, 1, input_ctrl_dim))
            
        self.range_info = range_info
        
        # lower and upper bound for stability
        if range_info:
            self.range_min = torch.Tensor([range_info[key][0] * 0.1 for key in range_info.keys()])
            self.range_max = torch.Tensor([range_info[key][1] * 10.0 for key in range_info.keys()])
        else:
            self.range_min = None
            self.range_max = None
        
    def forward(self, x_0D : torch.Tensor, x_ctrl : torch.Tensor, target_0D : Optional[torch.Tensor] = None, target_ctrl : Optional[torch.Tensor] = None):
        
        b = x_0D.size()[0]
        
        # add noise to robust performance
        x_0D = self.noise(x_0D)
        
        if self.RIN:
            means_0D = x_0D.mean(1, keepdim=True).detach()
            x_0D = x_0D - means_0D
            stdev_0D = torch.sqrt(torch.var(x_0D, dim=1, keepdim=True, unbiased=False) + 1e-3)
            x_0D /= stdev_0D
            x_0D = x_0D * self.affine_weight_0D + self.affine_bias_0D
            
            means_ctrl = x_ctrl.mean(1, keepdim=True).detach()
            x_ctrl = x_ctrl - means_ctrl
            stdev_ctrl = torch.sqrt(torch.var(x_ctrl, dim=1, keepdim=True, unbiased=False) + 1e-3)
            x_ctrl /= stdev_ctrl
            x_ctrl = x_ctrl * self.affine_weight_ctrl + self.affine_bias_ctrl
            
        # concat : (N, T, F1 + F2)
        x = torch.concat([x_0D, x_ctrl], axis = 2)
        
        # encoding : (N, T, F) -> (N, T, d_model)
        x = self.encoder_input(x)
        
        # (T, N, d_model)
        x = x.permute(1,0,2)
        
        if self.src_mask is None or self.src_mask.size()[0] != x.size()[0]:
            device = x.device
            mask = self._generate_square_subsequent_mask(x.size()[0], x.size()[0]).to(device)
            self.src_mask = mask
        
        # positional encoding for time axis : (T, N, d_model)
        x = self.pos(x)
        
        # transformer encoding layer : (T, N, d_model)
        x_enc = self.trans_enc(x, self.src_mask.to(x.device))
        
        # Decoder process
        target_0D = self.noise(target_0D)
        target = torch.concat([target_0D, target_ctrl], axis = 2)
        x_dec = self.decoder_input(target)
        x_dec = x_dec.permute(1,0,2)
        
        if self.tgt_dec_mask is None or self.src_dec_mask is None or self.tgt_dec_mask.size()[0] != x_dec.size()[0]:
            device = x_dec.device
            self.src_dec_mask = self._generate_square_subsequent_mask(x_dec.size()[0], x_enc.size()[0]).to(device)
            self.tgt_dec_mask = self._generate_square_subsequent_mask(x_dec.size()[0], x_dec.size()[0]).to(device)
        
        # x_enc : (T, N, d_model)
        # x_dec : (T',N, d_model)
        x_dec = self.trans_dec(
            tgt = x_dec,
            memory = x_enc,
            tgt_mask = self.tgt_dec_mask.to(x_dec.device),
            memory_mask = self.src_dec_mask.to(x_dec.device)
        )
        
        x_dec = x_dec.permute(1,0,2)
        x = self.lc_feat(x_dec)
        
        # RevIN for considering data distribution shift
        if self.RIN:
            x = x - self.affine_bias_0D
            x = x / (self.affine_weight_0D + 1e-3)
            x = x * stdev_0D
            x = x + means_0D  
        
        # clamping : output range
        x = torch.clamp(x, min = -10.0, max = 10.0)
        
        # remove nan value for stability
        x = torch.nan_to_num(x, nan = 0)
        return x

    def _generate_square_subsequent_mask(self, dim1 : int, dim2 : int):
        mask = (torch.triu(torch.ones(dim2, dim1))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def summary(self):
        sample_0D = torch.zeros((1, self.input_seq_len, self.input_0D_dim))
        sample_ctrl = torch.zeros((1, self.input_seq_len, self.input_ctrl_dim))
        sample_tgt_0D = torch.zeros((1, self.output_pred_len, self.input_0D_dim))
        sample_tgt_ctrl = torch.zeros((1, self.output_pred_len, self.input_ctrl_dim))
        summary(self, sample_0D, sample_ctrl, sample_tgt_0D, sample_tgt_ctrl, batch_size = 1, show_input = True, print_summary=True)