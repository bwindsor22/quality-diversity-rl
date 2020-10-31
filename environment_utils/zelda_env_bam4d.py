import gym
from gym import spaces
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
import pickle

ac = ["ACTION_NIL",
      "ACTION_UP",
      "ACTION_LEFT",
      "ACTION_DOWN",
      "ACTION_RIGHT",
      "ACTION_USE"]
direction = {2:3, 3:2, 4:1, 1:0}
direction_list = [1, 2, 3, 4]
Zelda = 7
Pacman = 27
solarfox = 1
class ZeldaEnv(gym.Wrapper):
    def __init__(self, env, crop=False, rotate=False, full=False, repava=False, train=False, shape=(84,84)):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.direction = 4
        self.crop = crop
        self.rotate = rotate
        self.full = full
        self.repava = repava
        self.observation_space = spaces.Box(low=0, high=255, shape=shape+(3,), dtype=np.uint8)
        self.shape = shape
        self.train = train
        self.init = None
        self.original = None
        self.done = False
        game_id = env.spec.id.split("-")
        game_id[2] = "lvl{}"
        self.game_id = "-".join(game_id)
        if self.crop:
            print("translate")
        if self.rotate:
            print("rotate")
        if self.full:
            print("full view")
        if self.repava:
            print("replace avatar")
        if self.train:
            print('training')
        # self.env.observation_space.shape = self.env.observation_space.shape[:-1] + (3,)

    def step(self, action):
        # restore actually action when performing rotate
        tuple_obs = None
        if action != 0 and action != 5:
            if self.rotate:
                action = direction_list[(direction_list.index(action)+4-direction.get(self.direction))%4]
            self.direction = action
#             print(self.direction)
        tuple_obs, reward, done, info = self.env.step(action)
            
        #if done:
            #if info.get("winner") == 3:
                #info["episode"]['c'] = 1
            #else:
                #info["episode"]['c'] = 0
        ascii_obs = "\n".join([",".join(["avatar" if tuple_obs[1][i,j,7]==1 else "" for j in range(tuple_obs[1].shape[1])]) for i in range(tuple_obs[1].shape[0])])
        obs = tuple_obs[0]
        self.original = copy.deepcopy(obs)
        info["ascii"] = tuple_obs[1]
        info['original'] = obs
#         print("rotation", direction.get(self.direction))
        if self.repava:
            try:
                obs = repava(obs, ascii_obs)
            except Exception as error:
                with open("solarfox_ascii", "wb") as f:
                    pickle.load(tuple_obs[1], f)
                cv2.imwrite("solarfox.jpg", tuple_obs[0])
                raise error
        if self.crop:
            obs = mask(obs, ascii_obs, direction.get(self.direction), rotate=self.rotate)
        elif self.full:
            obs = padded_view(obs, ascii_obs) if not self.rotate else padded_view(obs, ascii_obs, k=direction.get(self.direction))
        elif self.rotate:
            obs = np.rot90(obs, k=direction.get(self.direction))
            
        obs = cv2.resize(obs, self.shape)
        self.done = done
        return obs, reward, done, info

    def get_action_meanings(self):    
        return self.unwrapped.get_action_meanings()

    def reset(self):
        # level = '/home/chang/situated-zelda-test-2/GVGAI_GYM/gym_gvgai/envs/games/zelda_v2/zelda_lvl{}.txt'.format(idx)
#         print(self.game_id)
        tuple_obs = None
        level_data = None
        if self.train:
            level = np.random.randint(0,5)
#             print(level)
            tuple_obs = self.env.reset(environment_id=self.game_id.format(level))
#             level = np.random.choice([10,20,30,40,50])
# #             print(level)
#             with open("/home/chang/boulderdash_levels/lvl_{}.txt".format(level)) as f:
#                 level_data = "".join(f.readlines())
#             tuple_obs = self.env.reset(level_data=level_data)
        else:
#             print("testing")
            tuple_obs = self.env.reset()
        #d = np.random.randint(1, 5)
        d = 3
        self.direction = d
        if self.direction == 4:
            tuple_obs, _, _, info = self.env.step(0)
        else:
            tuple_obs, _, _, info = self.env.step(self.direction)
        tuple_obs, _, _, info = self.env.step(0)
        self.original = copy.deepcopy(tuple_obs[0])
        #Replacing 1 in tuple with 7 **Note Seems to Work
        #ascii_obs = "\n".join([",".join(["avatar" if tuple_obs[1][i,j,1]==1 else "" for j in range(tuple_obs[1].shape[1])]) for i in range(tuple_obs[1].shape[0])])
        ascii_obs = "\n".join([",".join(["avatar" if tuple_obs[1][i,j,7]==1 else "" for j in range(tuple_obs[1].shape[1])]) for i in range(tuple_obs[1].shape[0])])
        obs = tuple_obs[0]
        info["ascii"] = tuple_obs[1]
        if self.repava:
            obs = repava(obs, ascii_obs)
        if self.crop or self.full:
            if self.crop:
                print(ascii_obs)
                print(tuple_obs[1][1])
                #print(tuple_obs[1].shape)
                #print(obs.shape)
                obs = mask(obs, ascii_obs, direction.get(self.direction), rotate=self.rotate)
            elif self.full:
                obs = padded_view(obs, ascii_obs) if not self.rotate else padded_view(obs, ascii_obs, k=direction.get(self.direction))
        elif self.rotate:
            obs = np.rot90(obs, k=direction.get(self.direction))
        self.init = obs
        obs = cv2.resize(obs, self.shape)
        return obs
    
    
    def set_level(self, level_path):
        level_data = None
        with open(level_path) as f:
            level_data = "".join(f.readlines())
        self.env.reset(level_data=level_data)

def repava(image, ascii_obs, pixel=10):
    ascii = [l.split(",") for l in ascii_obs.split("\n")]
#     loc = np.asarray(np.where((np.core.defchararray.find(ascii,"nokey")!=-1)|(np.core.defchararray.find(ascii,"withkey")!=-1))).T[0]
    loc = np.asarray(np.where((np.core.defchararray.find(ascii,"avatar")!=-1))).T[0]
    image[pixel*loc[0]:pixel*(loc[0]+1), pixel*loc[1]:pixel*(loc[1]+1),:] = (255, 192, 203)
    return image
    

def crop(image, mask, ascii, u, d, l, r, pixel):
    # print("crop")
    # loc = np.asarray(np.where((ascii=="nokey")|(ascii=="withkey"))).T[0]
#     loc = np.asarray(np.where((np.core.defchararray.find(ascii,"nokey")!=-1)|(np.core.defchararray.find(ascii,"withkey")!=-1))).T[0]
    loc = np.asarray(np.where((np.core.defchararray.find(ascii,"avatar")!=-1))).T[0]
    blank = np.full(((u+d+1)*pixel, (l+r+1)*pixel, image.shape[2]), 0, dtype='uint8')
    for i in range(-u, d+1):
        for j in range(-l, r+1):
            if loc[0] + i >= 0 and loc[1] + j >= 0 and loc[0] + i <= ascii.shape[0] - 1  and loc[1] + j <= ascii.shape[1] - 1 and mask[i+u,j+l] != 'b':
                # pos on mask
                pos = [i+u, j+l]
                blank[pos[0]*pixel:(1+pos[0])*pixel, pos[1]*pixel:(pos[1]+1)*pixel, :] = image[(loc[0]+i)*pixel:(loc[0]+i+1)*pixel,(loc[1]+j)*pixel:(loc[1]+j+1)*pixel,:]
    return blank


def mask(image, ascii_obs, k=0, pixel=10, rotate=True):
#     mask = 's,s,s,s,s\ns,s,s,s,s\ns,s,a,s,s\ns,s,s,s,s\ns,s,s,s,s'
    mask = 's,s,s,s,s,s,s\ns,s,s,s,s,s,s\ns,s,s,s,s,s,s\ns,s,s,a,s,s,s\ns,s,s,s,s,s,s\ns,s,s,s,s,s,s\ns,s,s,s,s,s,s'
#     mask = 's,s,s,s,s\nb,s,s,s,b\nb,s,a,s,b\nb,b,s,b,b'
#     mask = 's,s,s,s,s,s,s\nb,s,s,s,s,s,b\nb,b,s,s,s,b,b\nb,b,s,a,s,b,b\nb,b,b,s,b,b,b'
    mask_list = np.array([l.split(",") for l in mask.split("\n")])
    mask_pos = np.asarray(np.where(mask_list=='a')).T[0]
    ascii = np.rot90([l.split(",") for l in ascii_obs.split("\n")], k=k)
    image = np.rot90(image, k=k)
    obs = crop(image, mask_list, ascii, mask_pos[0], mask_list.shape[0]-mask_pos[0]-1, mask_pos[1], mask_list.shape[1]-mask_pos[1]-1, pixel=pixel)
    return np.rot90(obs, k=4-k) if not rotate else obs

def padded_view(image, ascii_obs, k=0, pixel=10):
    image = np.rot90(image, k=k)
    ascii = np.rot90([l.split(",") for l in ascii_obs.split("\n")], k=k)
    h = w = int(max(image.shape)/pixel)
    padding = np.full(((2*h-3)*pixel, (2*w-3)*pixel, image.shape[2]), 0, dtype='uint8')
#     loc = np.asarray(np.where((np.core.defchararray.find(ascii,"nokey")!=-1)|(np.core.defchararray.find(ascii,"withkey")!=-1))).T[0]
    loc = np.asarray(np.where((np.core.defchararray.find(ascii,"avatar")!=-1))).T[0]
    center = (int((padding.shape[0]/pixel-1)/2), int((padding.shape[1]/pixel-1)/2))
    padding[pixel*(center[0]-loc[0]):pixel*(center[0]-loc[0]+int(image.shape[0]/pixel)),pixel*(center[1]-loc[1]):pixel*(center[1]-loc[1]+int(image.shape[1]/pixel)), :] = image
    return padding
