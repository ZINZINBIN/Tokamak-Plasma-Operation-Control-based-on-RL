import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch

resize = T.Compose([
    T.ToPILImage(),
    T.Resize(128, interpolation=Image.CUBIC),
    T.ToTensor()
])

EPS_START_DEFAULT = 0.9
EPS_END_DEFAULT = 0.05
EPS_DECAY_DEFAULT = 200

# image version enviornment
def get_screen(env):
    screen = env.render(mode = 'rgb_array').transpose((2,0,1))
    _, screen_height, screen_width = screen.shape

    # continous한 memory 형태로 반환
    screen = np.ascontiguousarray(screen, dtype = np.float32)
    screen = torch.from_numpy(screen)

    return resize(screen).unsqueeze(0)

# rendering from environment
# output : (1, seq_len, col_dim)
def get_state(env):
    pass