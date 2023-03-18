''' SCINet : Time Series is a Special Sequence, Forecasting with Sample Convolution and Interaction
    Reference
    - code repository : https://github.com/cure-lab/SCINet/blob/main/models/SCINet.py 
    - explaination : https://themore-dont-know.tistory.com/13 
'''

import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
from pytorch_model_summary import summary

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x:torch.Tensor):
        return x[:, ::2, :]

    def odd(self, x:torch.Tensor):
        return x[:, 1::2, :]

    def forward(self, x:torch.Tensor):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))
    
class Interactor(nn.Module):
    def __init__(self, in_planes:int, splitting:bool=True, kernel:int=5, dropout:float=0.5, groups:int= 1, hidden_size:int = 1, INN:bool=True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1 #by default: stride==1 
            pad_r = self.dilation * (self.kernel_size) // 2 + 1 #by default: stride==1 
        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
            
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        # module psi, phi, rho, etta
        # Interactive learning : split input as odd and even order and then compute interative learning
        
        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden), kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden), kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden), kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden), kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x:torch.Tensor):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))

            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)

            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)
        
class InteractorLevel(nn.Module):
    def __init__(self, in_planes:int, kernel:int, dropout:float, groups:int, hidden_size:int, INN:bool):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes = in_planes, splitting=True, kernel = kernel, dropout=dropout, groups = groups, hidden_size = hidden_size, INN = INN)
    def forward(self, x:torch.Tensor):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)

class LevelSCINet(nn.Module):
    def __init__(self, in_planes:int, kernel_size:int, dropout:float, groups:int, hidden_size:int, INN:bool):
        super(LevelSCINet, self).__init__()
        self.interact = InteractorLevel(in_planes= in_planes, kernel = kernel_size, dropout = dropout, groups =groups , hidden_size = hidden_size, INN = INN)

    def forward(self, x:torch.Tensor):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1) #even: B, T, D odd: B, T, D
    
# SCINet tree : binary tree architecture
# recursive binary tree architecture for even and odd

class SCINet_Tree(nn.Module):
    def __init__(self, in_planes:int, current_level:int, kernel_size:int, dropout:float, groups:int, hidden_size:int, INN:bool):
        super().__init__()
        # info about level
        self.current_level = current_level
        
        # SCIBlock for each node
        self.workingblock = LevelSCINet(
            in_planes = in_planes,
            kernel_size = kernel_size,
            dropout = dropout,
            groups= groups,
            hidden_size = hidden_size,
            INN = INN
        )

        # generate the child node 
        if current_level!=0:
            self.SCINet_Tree_odd=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size, INN)
            self.SCINet_Tree_even=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size, INN)
    
    def zip_up_the_pants(self, even:torch.Tensor, odd:torch.Tensor):
        # concatenate even and odd tensor with considering order
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2) #L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
            
        if odd_len < even_len: 
            _.append(even[-1].unsqueeze(0))
            
        return torch.cat(_,0).permute(1,0,2) #B, L, D
        
    def forward(self, x:torch.Tensor):
        # interactive learning : input -> (even, odd)
        x_even_update, x_odd_update= self.workingblock(x)
        
        # We recursively reordered these sub-series. You can run the ./utils/recursive_demo.py to emulate this procedure. 
        if self.current_level ==0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))
        
class EncoderTree(nn.Module):
    def __init__(self, in_planes:int,  num_levels:int, kernel_size:int, dropout:float, groups:int, hidden_size:int, INN:bool):
        super().__init__()
        self.levels=num_levels
        self.SCINet_Tree = SCINet_Tree(
            in_planes = in_planes,
            current_level = num_levels-1,
            kernel_size = kernel_size,
            dropout =dropout ,
            groups = groups,
            hidden_size = hidden_size,
            INN = INN
        )
        
    def forward(self, x:torch.Tensor):
        x= self.SCINet_Tree(x)
        return x

class SCINet(nn.Module):
    def __init__(
        self, 
        output_len:int, input_len:int, output_dim : int, input_dim:int = 9, hid_size:int = 1, num_stacks:int = 1,
        num_levels:int = 3, num_decoder_layer:int = 1, concat_len:int = 0, groups:int = 1, kernel:int = 5, dropout:float = 0.5,
        single_step_output_One:int = 0, input_len_seg:int = 0, positionalE:bool = False, modified:bool = True, RIN:bool = False
        ):
        super(SCINet, self).__init__()

        self.input_dim = input_dim # number of features for input data
        self.input_len = input_len # sequence length of input
        self.output_len = output_len # sequence length of output 
        self.output_dim = output_dim # number of features for output data
        self.hidden_size = hid_size 
        self.num_levels = num_levels # number of layers of SCINet tree
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE
        self.RIN=RIN # Reversible Instance Normalization
        self.num_decoder_layer = num_decoder_layer

        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified)

        if num_stacks == 2: # we only implement two stacks at most.
            self.blocks2 = EncoderTree(
                in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified)

        self.stacks = num_stacks # number of SCINet for stacking

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        self.div_projection = nn.ModuleList()
        self.overlap_len = self.input_len//4
        self.div_len = self.input_len//6

        if self.num_decoder_layer > 1:
            self.projection1 = nn.Linear(self.input_len, self.output_len)
            for layer_idx in range(self.num_decoder_layer-1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = min(i*self.div_len+self.overlap_len,self.input_len) - i*self.div_len
                    div_projection.append(nn.Linear(lens, self.div_len))
                self.div_projection.append(div_projection)

        if self.single_step_output_One: # only output the N_th timestep.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, 1, kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, 1, kernel_size = 1, bias = False)
        else: # output the N timesteps.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)

        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
    
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))
            
        ### Dimension reduction ###
        if self.input_dim != self.output_dim:
            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, (self.output_dim + self.input_dim)//2),
                nn.ReLU(),
                nn.Linear((self.output_dim + self.input_dim)//2, self.output_dim)
            )
        else:
            self.fc = None
    
    def get_position_encoding(self, x:torch.Tensor):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
    
        return signal

    def forward(self, x:torch.Tensor):
        
        assert self.input_len % (np.power(2, self.num_levels)) == 0 # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        ### activated when RIN flag is set ###
        if self.RIN:
            # print('/// RIN ACTIVATED ///\r',end='')
            means = x.mean(1, keepdim=True).detach()
            #mean
            x = x - means
            #var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x = x * self.affine_weight + self.affine_bias

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        if self.num_decoder_layer == 1:
            x = self.projection1(x)
        else:
            x = x.permute(0,2,1)
            for div_projection in self.div_projection:
                output = torch.zeros(x.shape,dtype=x.dtype).cuda()
                for i, div_layer in enumerate(div_projection):
                    div_x = x[:,:,i*self.div_len:min(i*self.div_len+self.overlap_len,self.input_len)]
                    output[:,:,i*self.div_len:(i+1)*self.div_len] = div_layer(div_x)
                x = output
            x = self.projection1(x)
            x = x.permute(0,2,1)

        if self.stacks == 1:
            ### reverse RIN ###
            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            if self.fc:
                x = self.fc(x)
            
            return x

        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)

            # the second stack
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)
            
            ### Reverse RIN ###
            if self.RIN:
                MidOutPut = MidOutPut - self.affine_bias
                MidOutPut = MidOutPut / (self.affine_weight + 1e-10)
                MidOutPut = MidOutPut * stdev
                MidOutPut = MidOutPut + means

            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means
                
            if self.fc:
                x = self.fc(x)

            return x, MidOutPut
        
    def summary(self):
        sample_data = torch.zeros((1, self.input_len, self.input_dim))
        return summary(self, sample_data, batch_size = 1, show_input = True, show_hierarchical=False, print_summary=True)
    
 
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
   
class SimpleSCINet(nn.Module):
    def __init__(
        self, 
        output_len:int, 
        input_len:int, 
        output_dim : int, 
        input_0D_dim : int = 5,
        input_ctrl_dim : int = 14,
        hid_size:int = 1, 
        num_levels:int = 3, 
        num_decoder_layer:int = 1, 
        concat_len:int = 0, 
        groups:int = 1, 
        kernel:int = 5, 
        dropout:float = 0.5,
        single_step_output_One:int = 0, 
        positionalE:bool = False, 
        modified:bool = True, 
        RIN:bool = False,
        noise_mean : float = 0,
        noise_std : float = 1.0,
        ):
        
        super(SimpleSCINet, self).__init__()

        self.input_0D_dim = input_0D_dim
        self.input_ctrl_dim = input_ctrl_dim
        
        self.input_dim = input_0D_dim + input_ctrl_dim # number of features for input data
        self.input_len = input_len # sequence length of input
        self.output_len = output_len # sequence length of output 
        self.output_dim = output_dim # number of features for output data
        self.hidden_size = hid_size 
        self.num_levels = num_levels # number of layers of SCINet tree
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE
        self.RIN=RIN # Reversible Instance Normalization
        self.num_decoder_layer = num_decoder_layer
        
        self.noise_layer = NoiseLayer(noise_mean, noise_std)

        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        self.div_projection = nn.ModuleList()
        self.overlap_len = self.input_len//4
        self.div_len = self.input_len//6

        if self.num_decoder_layer > 1:
            self.projection1 = nn.Linear(self.input_len, self.output_len)
            for layer_idx in range(self.num_decoder_layer-1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = min(i*self.div_len+self.overlap_len,self.input_len) - i*self.div_len
                    div_projection.append(nn.Linear(lens, self.div_len))
                self.div_projection.append(div_projection)

        # For positional encoding
        self.pe_hidden_size = self.input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
    
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))

        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float32) * (-1) * log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.input_dim))
            
        ### Dimension reduction ###
        if self.input_dim != self.output_dim:
            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, (self.output_dim + self.input_dim)//2),
                nn.ReLU(),
                nn.Linear((self.output_dim + self.input_dim)//2, self.output_dim)
            )
        else:
            self.fc = None
    
    def get_position_encoding(self, x:torch.Tensor):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
    
        return signal

    def forward(self, x_0D: torch.Tensor, x_ctrl : torch.Tensor):
        
        assert self.input_len % (np.power(2, self.num_levels)) == 0 # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        
        # modify sequence length of x_ctrl
        if x_ctrl.size()[1] != self.input_len:
            x_ctrl = x_ctrl[:,:self.input_len,:]
        
        # add noise
        x_0D = self.noise_layer(x_0D)
        x = torch.concat([x_0D, x_ctrl], axis = 2)
    
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        ### activated when RIN flag is set ###
        if self.RIN:
            # print('/// RIN ACTIVATED ///\r',end='')
            means = x.mean(1, keepdim=True).detach()
            #mean
            x = x - means
            #var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x = x * self.affine_weight + self.affine_bias

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        if self.num_decoder_layer == 1:
            x = self.projection1(x)
        else:
            x = x.permute(0,2,1)
            for div_projection in self.div_projection:
                output = torch.zeros(x.shape,dtype=x.dtype).cuda()
                for i, div_layer in enumerate(div_projection):
                    div_x = x[:,:,i*self.div_len:min(i*self.div_len+self.overlap_len,self.input_len)]
                    output[:,:,i*self.div_len:(i+1)*self.div_len] = div_layer(div_x)
                x = output
            x = self.projection1(x)
            x = x.permute(0,2,1)

        ### reverse RIN ###
        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means

        if self.fc:
            x = self.fc(x)
            
        # clamping : output range
        x = torch.clamp(x, min = -10.0, max = 10.0)
        
        # remove nan value for stability
        x = torch.nan_to_num(x, nan = 0)
            
        return x
        
    def summary(self):
        sample_0D = torch.zeros((1, self.input_len, self.input_0D_dim))
        sample_ctrl = torch.zeros((1, self.input_len, self.input_ctrl_dim))
        
        return summary(self, sample_0D, sample_ctrl, batch_size = 1, show_input = True, show_hierarchical=False, print_summary=True)