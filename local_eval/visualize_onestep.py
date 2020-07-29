import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path


prefix = '/Users/bradwindsor/ms_projects/qd-gen/gameQD/saves_numpy/'
run_name = 'gvgai-zelda-lvl4-v0_step_337_act_1_reward_2.0_crit_other'
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
