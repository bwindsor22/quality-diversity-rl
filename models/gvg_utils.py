import numpy as np
import torch
import torchvision.transforms as T

resize = T.Compose([T.ToPILImage(),
                    T.Resize(30),
                    T.ToTensor()])

def get_screen(env, device):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    # (this doesn't require a copy)
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)
