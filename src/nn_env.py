import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from typing import List, Optional, Union, Tuple
from pytorch_model_summary import summary

class ConvLSTM(nn.Module):
    def __init__(
        self, 
        seq_len : int = 21, 
        pred_len : int = 7,
        col_dim : int = 10, 
        conv_dim : int = 32, 
        conv_kernel : int = 3,
        conv_stride : int = 1, 
        conv_padding : int = 1,
        mlp_dim : int = 64,
        output_dim : int = 1,
        ):
        
        super(ConvLSTM, self).__init__()
        self.col_dim = col_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.lstm_dim = pred_len
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim

        # spatio-conv encoder : analyze spatio-effect between variables
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = col_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim), 
            nn.ReLU(),
            nn.Conv1d(in_channels = conv_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim), 
            nn.ReLU(),
        )

        lstm_input_dim = self.compute_conv1d_output_dim(self.compute_conv1d_output_dim(seq_len, conv_kernel, conv_stride, conv_padding, 1), conv_kernel, conv_stride, conv_padding, 1)

        # temporl - lstm
        self.lstm = nn.LSTM(lstm_input_dim, self.lstm_dim, bidirectional = True, batch_first = False)
        self.w_s1 = nn.Linear(self.lstm_dim * 2, self.lstm_dim)
        self.w_s2 = nn.Linear(self.lstm_dim, self.lstm_dim)

        self.regressor = nn.Sequential(
            nn.Linear(self.lstm_dim * 2, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim)
        )
    
    def compute_conv1d_output_dim(self, input_dim : int, kernel_size : int = 3, stride : int = 1, padding : int = 1, dilation : int = 1):
        return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def attention(self, lstm_output : torch.Tensor):
        attn_weight_matrix = self.w_s2(torch.tanh(self.w_s1(lstm_output)))
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim = 2)
        return attn_weight_matrix
    
    def forward(self, x : torch.Tensor):
        
        # x : (batch, seq_len, col_dim)
        x_conv = self.conv(x.permute(0,2,1))
        h_0 = Variable(torch.zeros(2, x.size()[0], self.lstm_dim)).to(x.device)
        c_0 = Variable(torch.zeros(2, x.size()[0], self.lstm_dim)).to(x.device)

        lstm_output, (h_n,c_n) = self.lstm(x_conv.permute(1,0,2), (h_0, c_0))
        lstm_output = lstm_output.permute(1,0,2)
        
        att = self.attention(lstm_output)
        hidden = torch.bmm(att.permute(0,2,1), lstm_output)
        hidden = hidden.view(hidden.size()[0], self.pred_len, -1)
        output = self.regressor(hidden)
        return output
    
    def summary(self, device : str = 'cpu', show_input : bool = True, show_hierarchical : bool = False, print_summary : bool = True, show_parent_layers : bool = False):
        sample = torch.zeros((1, self.seq_len, self.col_dim), device = device)
        return summary(self, sample, show_input = show_input, show_hierarchical=show_hierarchical, print_summary = print_summary, show_parent_layers=show_parent_layers)