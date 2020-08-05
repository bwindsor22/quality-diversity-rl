import numpy as np
import torch
import matplotlib.pyplot as plt

name1 = '7bb7df17-0bba-4010-87ac-c33f4cdfa79c_gvgai-zelda-lvl3-v0_step_103_act_1_reward_0.0_crit_other_seq_0'
name2 = '7bb7df17-0bba-4010-87ac-c33f4cdfa79c_gvgai-zelda-lvl3-v0_step_103_act_1_reward_0.0_crit_other_seq_1'
prefix = '/Users/bradwindsor/ms_projects/qd-gen/gameQD/saves_numpy/'
suffix = '.npy'

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def reshape(arry):
    return torch.tensor(arry).cpu().squeeze(0).permute(1, 2, 0).numpy()


f1 = np.load(open(prefix + name1 + suffix, 'rb'))
f2 = np.load(open(prefix + name2 + suffix, 'rb'))
r1 = reshape(f1)
r2 = reshape(f2)
diff = mse(r1, r2)
print('diff', diff)


for i, arry in enumerate([r1, r2]):
    good_image = arry
    plt.imsave(f'{i}.png', good_image)
