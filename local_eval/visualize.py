import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path

hpc_files = Path('/Users/bradwindsor/ms_projects/hpc_files/').glob('*')

for i, f in enumerate(hpc_files):
    try:
        a = np.load(str(f))
        t = torch.tensor(a)
        good_image = t.cpu().squeeze(0).permute(1, 2, 0).numpy()
        plt.imsave(f'{i}_{f.stem}.png', good_image)
    except Exception:
        pass