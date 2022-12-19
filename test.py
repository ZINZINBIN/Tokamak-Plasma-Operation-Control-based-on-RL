from src.rl.ddpg import Encoder
import torch

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:" + str(0)
else:
    device = 'cpu'
    
encoder = Encoder(9, 21, 32, 3, 2, 1)
encoder.to(device)

sample_data = torch.zeros((1, 21, 9)).to(device)
print(encoder(sample_data).size())
print(encoder.linear_input_dim)