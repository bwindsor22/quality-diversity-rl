import gym
from gym import spaces
import numpy as np
import cv2

ac = ["ACTION_NIL",
      "ACTION_USE",
      "ACTION_LEFT",
      "ACTION_RIGHT",
      "ACTION_DOWN",
      "ACTION_UP"]
direction = {2:3, 3:1, 4:2, 5:0}
direction_list = [5, 2, 4, 3]

class ZeldaEnv(gym.Wrapper):
    def __init__(self, env, crop=False, rotate=False, full=False, repava=False, shape=(84,84)):
        print(type(crop), type(rotate))
        gym.Wrapper.__init__(self, env)
        self.env = env
        # self.direction = 3
        self.crop = crop
        self.rotate = rotate
        self.full = full
        self.repava = repava
        self.observation_space = spaces.Box(low=0, high=255, shape=shape+(3,), dtype=self.env.observation_space.dtype)
        self.shape = shape
        if self.crop:
            print("translate")
        if self.rotate:
            print("rotate")
        if self.full:
            print("full view")
        if self.repava:
            print("replace avatar")
        # self.env.observation_space.shape = self.env.observation_space.shape[:-1] + (3,)

    def step(self, action):
        # restore actually action when performing rotate
        if action != 0 and action != 1:
            if self.rotate:
                action = direction_list[(direction_list.index(action)+4-direction.get(self.direction))%4]
            self.direction = action
        obs, reward, done, info = self.env.step(action)
        if done:
            if info.get("winner") == 'PLAYER_WINS':
                info["episode"]['c'] = 1
            else:
                info["episode"]['c'] = 0
        
        if self.crop:
            obs = mask(obs, info, direction.get(self.direction), rotate=self.rotate, repava=self.repava)
        elif self.full:
            obs = padded_view(obs, info) if not self.rotate else padded_view(obs, info, k=direction.get(self.direction), repava=self.repava)
        elif self.rotate:
            obs = np.rot90(obs, k=direction.get(self.direction))

        obs = cv2.resize(obs, self.shape)
        return obs[:,:,:-1], reward, done, info

    def get_action_meanings(self):    
        return self.unwrapped.get_action_meanings()

    def reset(self):
        idx = np.random.randint(0,1190)
        # level = '/home/chang/situated-zelda-test-2/GVGAI_GYM/gym_gvgai/envs/games/zelda_v2/zelda_lvl{}.txt'.format(idx)
        level = np.random.randint(0,5)
        self.unwrapped._setLevel(level)
        obs = self.env.reset()
        d = np.random.randint(2, 6)
        self.direction = d
        if self.direction == 3:
            obs, _, _, info = self.env.step(0)
        else:
            obs, _, _, info = self.env.step(self.direction)
        if self.crop or self.full:
            if self.crop:
                obs = mask(obs, info, direction.get(self.direction), rotate=self.rotate, repava=self.repava)
            elif self.full:
                obs = padded_view(obs, info) if not self.rotate else padded_view(obs, info, k=direction.get(self.direction), repava=self.repava)
        elif self.rotate:
            obs = np.rot90(obs, k=direction.get(self.direction))

        obs = cv2.resize(obs, self.shape)
        return obs[:,:,:-1]

def resize(image, shape):
    image = cv2.resize(image, shape)
    return image
    

def crop(image, mask, ascii, u, d, l, r, pixel, repava):
    # print("crop")
    # loc = np.asarray(np.where((ascii=="nokey")|(ascii=="withkey"))).T[0]
    loc = np.asarray(np.where((np.core.defchararray.find(ascii,"nokey")!=-1)|(np.core.defchararray.find(ascii,"withkey")!=-1))).T[0]

    blank = np.full(((u+d+1)*pixel, (l+r+1)*pixel, image.shape[2]), 0, dtype='uint8')
    for i in range(-u, d+1):
        for j in range(-l, r+1):
            if loc[0] + i >= 0 and loc[1] + j >= 0 and loc[0] + i <= ascii.shape[0] - 1  and loc[1] + j <= ascii.shape[1] - 1 and mask[i+u,j+l] != 'b':
                # pos on mask
                pos = [i+u, j+l]
                blank[pos[0]*pixel:(1+pos[0])*pixel, pos[1]*pixel:(pos[1]+1)*pixel, :] = image[(loc[0]+i)*pixel:(loc[0]+i+1)*pixel,(loc[1]+j)*pixel:(loc[1]+j+1)*pixel,:]
    if repava:
        blank[u*pixel:(1+u)*pixel, l*pixel:(l+1)*pixel, :] = (255, 192, 203, 0)
    return blank


def mask(image, info, k=0, pixel=10, rotate=True, repava=True):
    mask = 's,s,s,s,s\ns,s,s,s,s\ns,s,a,s,s\ns,s,s,s,s\ns,s,s,s,s'
#     mask = 's,s,s,s,s\nb,s,s,s,b\nb,s,a,s,b\nb,b,s,b,b'
    # mask = 's,s,s,s,s,s,s\nb,s,s,s,s,s,b\nb,b,s,s,s,b,b\nb,b,s,a,s,b,b\nb,b,b,s,b,b,b'
    mask_list = np.array([l.split(",") for l in mask.split("\n")])
    mask_pos = np.asarray(np.where(mask_list=='a')).T[0]
    ascii = np.rot90([l.split(",") for l in info["ascii"].split("\n")], k=k)
    image = np.rot90(image, k=k)
    obs = crop(image, mask_list, ascii, mask_pos[0], mask_list.shape[0]-mask_pos[0]-1, mask_pos[1], mask_list.shape[1]-mask_pos[1]-1, pixel=pixel, repava=repava)
    return np.rot90(obs, k=4-k) if not rotate else obs

def padded_view(image, info, k=0, pixel=10, repava=True):
    image = np.rot90(image, k=k)
    ascii = np.rot90([l.split(",") for l in info["ascii"].split("\n")], k=k)
    h = w = int(max(image.shape)/pixel)
    padding = np.full(((2*h-3)*pixel, (2*w-3)*pixel, image.shape[2]), 0, dtype='uint8')
    loc = np.asarray(np.where((np.core.defchararray.find(ascii,"nokey")!=-1)|(np.core.defchararray.find(ascii,"withkey")!=-1))).T[0]
    center = (int((padding.shape[0]/pixel-1)/2), int((padding.shape[1]/pixel-1)/2))
    padding[pixel*(center[0]-loc[0]):pixel*(center[0]-loc[0]+int(image.shape[0]/pixel)),pixel*(center[1]-loc[1]):pixel*(center[1]-loc[1]+int(image.shape[1]/pixel)), :] = image
    if repava:
        padding[pixel*(center[0]):pixel*(center[0]+1), pixel*(center[1]):pixel*(center[0]+1), :] = (255, 192, 203, 0)
    return padding
