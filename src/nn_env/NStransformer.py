'''
    Reference
    - code : https://github.com/thuml/Nonstationary_Transformers/blob/main/ns_models/ns_Transformer.py
    - Non-stationary transformer
'''
import torch, math
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional, Dict, List
from pytorch_model_summary import summary

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

# positional encoding process
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
    
# De-stationary attention module
class DSAttention(nn.Module):
    def __init__(self, mask_flag : bool = True, factor : float = 5, scale = None, dropout : float = 0.1, output_attention : bool = False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, att_mask : torch.Tensor, tau : Optional[torch.Tensor]= None, delta : Optional[torch.Tensor] = None):
        B, L, H, E = q.size()
        _, S, _, D = v.size()
        scale = self.scale or 1.0 / math.sqrt(E)
        
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
        
        scores = torch.einsum("blhe,bshe->bhls", q, k) * tau + delta
        
        if self.mask_flag:
            scores.masked_fill_(att_mask.bool(), -torch.inf)
        
        A = self.dropout(torch.softmax(scale * scores, dim = -1))
        V = torch.einsum("bhls,bshd->blhd", A, v)
        
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, attention : DSAttention, d_model : int, n_heads : int, d_keys : Optional[int] = None, d_values : Optional[int]= None): 
        super().__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.q_proj = nn.Linear(d_model, d_keys * n_heads)
        self.k_proj = nn.Linear(d_model, d_keys * n_heads)
        self.v_proj = nn.Linear(d_model, d_values * n_heads)
        self.out_proj = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    
    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, att_mask : torch.Tensor,tau : Optional[torch.Tensor]= None, delta : Optional[torch.Tensor] = None):
        B, L, _ = q.size()
        _, S, _ = k.size()
        H = self.n_heads
        
        q = self.q_proj(q).view(B, L, H, -1)
        k = self.k_proj(k).view(B, S, H, -1)
        v = self.v_proj(v).view(B, S, H, -1)
        
        out, att = self.inner_attention(q,k,v,att_mask,tau,delta)
        out = out.view(B,L,-1)
        out = self.out_proj(out)
        
        return out, att
        
class EncoderLayer(nn.Module):
    def __init__(self, attention : DSAttention, d_model : int, d_ff : int, dropout : float = 0.1):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        
    def forward(self, x : torch.Tensor, att_mask : torch.Tensor, tau : Optional[torch.Tensor]=None, delta : Optional[torch.Tensor] = None):
        x_n, attn = self.attention(x,x,x,att_mask,tau,delta)
        x = self.dropout(x_n) + x
        x_branch= self.norm1(x)
        out = self.dropout(self.activation(self.conv1(x_branch.transpose(-1,1))))
        out = self.dropout(self.conv2(out).transpose(-1,1))
    
        out = self.norm2(out + x_branch)

        return out, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers:List, norm_layer : Optional[nn.Module]=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
    
    def forward(self, x : torch.Tensor, attn_mask : torch.Tensor, tau : Optional[torch.Tensor]=None, delta : Optional[torch.Tensor] = None):
        # x : (B, L, D)
        attns = []
        
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask, tau, delta)
            attns.append(attn)
        
        if self.norm:
            x = self.norm(x)
        return x, attns
  
# Projector : MLP to learn De-stationary factors
class Projector(nn.Module):
    def __init__(self, enc_in : int, seq_len : int, hidden_dim : int, output_dim : int, kernel_size : int = 3):
        super().__init__()
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size = kernel_size, padding = padding, padding_mode='circular', bias = False)
        self.backbone = nn.Sequential(
            nn.Linear(2 * enc_in, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, output_dim, bias = False)
        )
    
    def forward(self, x : torch.Tensor, stats : torch.Tensor):
        # x : (B,S,E)
        # stats : (B,1,E)
        
        B = x.size()[0]
        # (B,1,E)
        x = self.series_conv(x)
        # (B, 2, E)
        x = torch.cat([x, stats], dim = 1)
        # (B, 2E)
        x = x.view(B, -1)
        # (B, output_dim)
        x = self.backbone(x)
        return x

class NStransformer(nn.Module):
    def __init__(
        self, 
        n_layers : int = 2, 
        n_heads : int = 8, 
        dim_feedforward : int = 1024, 
        dropout : float = 0.1,      
        factor : float = 5.0,  
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
        ):
        
        super(NStransformer, self).__init__()
        
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
    
        # Noise added for robust performance
        self.noise = NoiseLayer(mean = noise_mean, std = noise_std)
        
        # 0D data encoder
        self.encoder_input_0D = nn.Sequential(
            nn.Linear(in_features=input_0D_dim, out_features=feature_0D_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_0D_dim //2, feature_0D_dim)
        )
        
        self.pos_0D = PositionalEncoding(d_model = feature_0D_dim, max_len = input_0D_seq_len)
        
        self.trans_enc_0D = Encoder(
            [   
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(True, factor, None, dropout, True),
                        feature_0D_dim,
                        n_heads,
                        dim_feedforward
                    ),
                    feature_0D_dim,
                    dim_feedforward,
                    dropout
                ) for _ in range(n_layers)
            ],
            norm_layer=nn.LayerNorm(feature_0D_dim)
        )
        
        # ctrl data encoder
        self.encoder_input_ctrl = nn.Sequential(
            nn.Linear(in_features=input_ctrl_dim, out_features=feature_ctrl_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_ctrl_dim //2, feature_ctrl_dim)
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
        
        # Tau and Delta : De-stationary Learner
        self.tau_learner = Projector(input_0D_dim, input_0D_seq_len, feature_0D_dim, 1)
        self.delta_learner = Projector(input_0D_dim, input_0D_seq_len, feature_0D_dim, input_0D_seq_len)

        # FC decoder
        # sequence length reduction
        self.lc_0D_seq = nn.Linear(input_0D_seq_len, output_0D_pred_len)
        self.lc_ctrl_seq = nn.Linear(input_ctrl_seq_len, output_0D_pred_len)
        
        # dimension reduction
        self.lc_feat= nn.Linear(feature_0D_dim + feature_ctrl_dim, output_0D_dim)
         
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
        x_0D_origin = x_0D.clone().detach()
        
        # Normalization
        means_0D = x_0D.mean(1, keepdim=True).detach()
        x_0D = x_0D - means_0D
        stdev_0D = torch.sqrt(torch.var(x_0D, dim=1, keepdim=True, unbiased=False) + 1e-3).detach()
        x_0D /= stdev_0D

        # add noise to robust performance
        x_0D = self.noise(x_0D)
       
        # path : 0D data
        # encoding : (N, T, F) -> (N, T, d_model)
        x_0D = self.encoder_input_0D(x_0D)
        
        # (T, N, d_model)
        x_0D = x_0D.permute(1,0,2)
        
        if self.src_mask_0D is None or self.src_mask_0D.size(0) != len(x_0D):
            device = x_0D.device
            mask = self._generate_square_subsequent_mask(len(x_0D)).to(device)
            self.src_mask_0D = mask
        
        # positional encoding for time axis : (T, N, d_model)
        x_0D = self.pos_0D(x_0D)
        
        # tau and delta for non-stationarity
        tau = self.tau_learner(x_0D_origin, stdev_0D).exp()
        delta = self.delta_learner(x_0D_origin, means_0D)
        
        # transformer encoding layer : (T, N, d_model) -> (N, T, d_model)
        x_0D = x_0D.permute(1,0,2)
        x_0D, _ = self.trans_enc_0D(x_0D, self.src_mask_0D.to(x_0D.device), tau, delta)
        
        # (N, T, d_model)
        # x_0D = x_0D.permute(1,0,2)
        
        # path : ctrl data
        x_ctrl = self.encoder_input_ctrl(x_ctrl)
        
        # (T, N, d_model)
        x_ctrl = x_ctrl.permute(1,0,2)
        
        if self.src_mask_ctrl is None or self.src_mask_ctrl.size(0) != len(x_ctrl):
            device = x_ctrl.device
            mask = self._generate_square_subsequent_mask(len(x_ctrl)).to(device)
            self.src_mask_ctrl = mask
        
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
        
        # clamping : output range
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
        