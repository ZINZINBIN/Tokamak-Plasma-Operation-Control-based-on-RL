''' 
    Non-stationary Transformer (NeurIPS 2022)
    
    A novel Transformer architecture for covering the Non-stationary data with utilizing De-stationary Attention module
    and Series stationarization. One of the severe issue related to the Transformer is Over-Stationarization. The vanilla 
    attention module can not effectively capture the temporal features from the Non-stationary temporal data.
    
    In this code, we applied DS Attention module based Transformer model to simulate a virtual KSTAR environment.
    We referred the example code and paper as given below. 
    
    Reference
    - Paper: Non-stationary Transformer (https://arxiv.org/abs/2205.14415)
    - code : https://github.com/thuml/Nonstationary_Transformers/blob/main/ns_models/ns_Transformer.py
    - Non-stationary transformer
'''
import torch, math
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional, Dict, List
from pytorch_model_summary import summary
from src.nn_env.transformer import PositionalEncoding, NoiseLayer, GELU

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
        out = x = self.norm1(x)
        out = self.dropout(self.activation(self.conv1(out.transpose(-1,1))))
        out = self.dropout(self.conv2(out).transpose(-1,1))
        out = self.norm2(out + x)
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
    
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):

        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])  
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
  
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
        input_ctrl_dim : int = 14,
        input_seq_len : int = 16,
        output_pred_len : int = 4,
        output_0D_dim : int = 12,
        feature_dim : int = 128,
        range_info : Optional[Dict] = None,
        noise_mean : float = 0,
        noise_std : float = 0.81,
        kernel_size : int = 3,
        ):
        
        super(NStransformer, self).__init__()
        
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
    
        # Noise added for robust performance
        self.noise = NoiseLayer(mean = noise_mean, std = noise_std)
        
        if kernel_size // 2 == 0:
            print("kernel sholud be odd number")
            kernel_size += 1
        padding = (kernel_size - 1) // 2
        
        # 0D data encoder
        self.encoder_input = nn.Sequential(
            nn.Conv1d(in_channels=input_0D_dim + input_ctrl_dim, out_channels=feature_dim // 2, kernel_size= kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_dim//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim // 2, out_channels=feature_dim, kernel_size= kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        self.enc_pos = PositionalEncoding(d_model = feature_dim, max_len = input_seq_len)
        
        self.trans_enc = Encoder(
            [   
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(True, factor, None, dropout, True),
                        feature_dim,
                        n_heads,
                        dim_feedforward
                    ),
                    feature_dim,
                    dim_feedforward,
                    dropout
                ) for _ in range(n_layers)
            ],
            norm_layer=nn.LayerNorm(feature_dim)
        )

        # Tau and Delta : De-stationary Learner
        self.tau_learner = Projector(input_0D_dim, input_seq_len, feature_dim, 1)
        self.delta_learner = Projector(input_0D_dim, input_seq_len, feature_dim, input_seq_len)

        # decoder
        self.decoder_input = nn.Sequential(
            nn.Conv1d(in_channels=input_ctrl_dim, out_channels=feature_dim // 2, kernel_size= kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_dim//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim // 2, out_channels=feature_dim, kernel_size= kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        self.dec_pos = PositionalEncoding(d_model = feature_dim, max_len = input_seq_len)
        
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, factor, None, dropout, False),
                        feature_dim,
                        n_heads,
                        dim_feedforward
                    ),
                    AttentionLayer(
                        DSAttention(False, factor, None, dropout, False),
                        feature_dim,
                        n_heads,
                        dim_feedforward
                    ),
                    feature_dim,
                    dim_feedforward,
                    dropout
                ) for _ in range(n_layers)
            ],
            norm_layer=nn.LayerNorm(feature_dim),
            projection = nn.Linear(feature_dim, output_0D_dim)
        )
        
        self.range_info = range_info
        
        if range_info:
            self.range_min = torch.Tensor([range_info[key][0] * 0.1 for key in range_info.keys()])
            self.range_max = torch.Tensor([range_info[key][1] * 10.0 for key in range_info.keys()])
        else:
            self.range_min = None
            self.range_max = None
            
    def forward(self, x_0D : torch.Tensor, x_ctrl : torch.Tensor, target_0D : Optional[torch.Tensor] = None, target_ctrl : Optional[torch.Tensor] = None):
        
        b = x_0D.size()[0]
        x_0D_origin = x_0D.clone().detach()
        
        # Normalization
        means_0D = x_0D.mean(1, keepdim=True).detach()
        x_0D = x_0D - means_0D
        stdev_0D = torch.sqrt(torch.var(x_0D, dim=1, keepdim=True, unbiased=False) + 1e-3).detach()
        x_0D /= stdev_0D

        # add noise to robust performance
        x_0D = self.noise(x_0D)
        
        # concat : (N, T, F1 + F2)
        x = torch.concat([x_0D, x_ctrl], axis = 2)
        
        # encoding : (N, T, F) -> (N, T, d_model)
        x = self.encoder_input(x)
 
        # (T, N, d_model)
        x = x.permute(1,0,2)
        
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask
        
        # positional encoding for time axis : (T, N, d_model)
        x = self.enc_pos(x)
        
        # tau and delta for non-stationarity
        tau = self.tau_learner(x_0D_origin, stdev_0D).exp()
        delta = self.delta_learner(x_0D_origin, means_0D)
        
        # transformer encoding layer : (T, N, d_model) -> (N, T, d_model)
        # x_0D = x_0D.permute(1,0,2)
        x, _ = self.trans_enc(x, self.src_mask.to(x.device), tau, delta)
        
        # Decoder process
        target_0D = self.noise(target_0D)
        target = torch.concat([target_0D, target_ctrl], axis = 2)
        x_dec = self.decoder_input(target)
        x_dec = x_dec.permute(1,0,2)
        
        if self.tgt_dec_mask is None or self.tgt_dec_mask.size(0) != len(x_dec):
            device = x_dec.device
            mask = self._generate_square_subsequent_mask(len(x_dec)).to(device)
            self.tgt_dec_mask = mask
        
        # positional encoding for time axis : (T, N, d_model)
        x_dec = self.dec_pos(x_dec)
        
        # dim reduction
        x = self.decoder(x_dec, x, x_mask = self.tgt_dec_mask, cross_mask = None, tau = tau, delta = delta)
        
        x = x * stdev_0D + means_0D
        
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
        sample_0D = torch.zeros((1, self.input_seq_len, self.input_0D_dim))
        sample_ctrl = torch.zeros((1, self.input_seq_len, self.input_ctrl_dim))
        summary(self, sample_0D, sample_ctrl, batch_size = 1, show_input = True, print_summary=True)
        