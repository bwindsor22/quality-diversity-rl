import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path


prefix = '/Users/bradwindsor/ms_projects/qd-gen/gameQD/saves_numpy/'
run_name = '68b95170-202d-4f49-a50c-e9dc51d1149e'
files = Path(prefix).glob(f'{run_name}*')

for i, f in enumerate(files):
    print('f')
    try:
        a = np.load(str(f))
        t = torch.tensor(a)
        good_image = t.cpu().squeeze(0).permute(1, 2, 0).numpy()
        plt.imsave(f'{i}_{f.stem}.png', good_image)
    except Exception:
        print('exc')
        pass
