import torch, math
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional
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
        input_dim: int = 11, 
        feature_dim : int = 256, 
        seq_len : int = 128, 
        n_layers : int = 1, 
        n_heads : int = 8, 
        dim_feedforward : int = 1024, 
        dropout : float = 0.1, 
        pred_len : int = 7, 
        output_dim : int = 1,
        RIN : bool = False
        ):
        
        super(Transformer, self).__init__()
        
        self.src_mask = None
        self.input_dim = input_dim
        self.max_len = seq_len
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.RIN = RIN
        
        self.noise = NoiseLayer(mean = 0, std = 1e-2)
        
        self.encoder_input_layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim //2, feature_dim)
        )
        
        self.pos_enc = PositionalEncoding(d_model = feature_dim, max_len = seq_len)
        
        self.encoder = nn.TransformerEncoderLayer(
            d_model = feature_dim, 
            nhead = n_heads, 
            dropout = dropout,
            dim_feedforward = dim_feedforward,
            activation = GELU()
        )
        
        # encoder layer
        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_layers=n_layers)
        
        # FC decoder
        # dimension reduction
        self.lc_feat= nn.Linear(feature_dim, input_dim)
        
        # sequence length reduction
        self.lc_seq = nn.Linear(seq_len, pred_len)
        
        if input_dim != output_dim:
            self.lc_feat2 = nn.Linear(input_dim, output_dim)
        
        # Reversible Instance Normalization
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))
        
    def forward(self, x : torch.Tensor):
        
        b = x.size()[0]
        
        # add noise to robust performance
        x = self.noise(x)
        
        if self.RIN:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            x = x * self.affine_weight + self.affine_bias
        
        # encoding : (N, T, F) -> (N, T, d_model)
        x = self.encoder_input_layer(x)
        
        # (T, N, d_model)
        x = x.permute(1,0,2)
        
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask
        
        # positional encoding for time axis : (T, N, d_model)
        x = self.pos_enc(x)
        
        # transformer encoding layer : (T, N, d_model)
        x = self.transformer_encoder(x, self.src_mask.to(x.device))
        
        # (N, T, d_model)
        x = x.permute(1,0,2)
        
        # dim reduction
        x = self.lc_feat(x)
        
        # seq reduction
        x = x.permute(0,2,1)
        x = self.lc_seq(x)
        x = x.permute(0,2,1)
        
        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means
            
        if self.lc_feat2:
            x = self.lc_feat2(x)
        
        return x

    def _generate_square_subsequent_mask(self, size : int):
        mask = (torch.triu(torch.ones(size,size))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def summary(self):
        sample_x = torch.zeros((1, self.max_len, self.input_dim))
        summary(self, sample_x, batch_size = 1, show_input = True, print_summary=True)
        
    def predict(self, x : torch.Tensor, target_len : Optional[int]):
        
        output = torch.zeros((self.max_len + target_len, self.n_features,)).to(x.device)
        output[0:self.max_len,:] = x
        
        if x.ndim == 2:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            for idx_srt in range(0, self.max_len + target_len, self.pred_len):
                
                if self.max_len + target_len - idx_srt < self.pred_len:
                    idx_end = -1
                    idx_srt = idx_end - self.max_len
                else:
                    idx_end = idx_srt + self.pred_len
                
                pred = self.forward(x).squeeze(0)
                output[idx_srt : idx_end,:] = pred
                
        output = output[self.max_len:, :].unsqueeze(0)
        return output
            