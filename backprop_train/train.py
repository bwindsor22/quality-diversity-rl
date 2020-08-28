import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging

logging.basicConfig(filename='../logs/parent-{}.log'.format('train'), level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info('Initializing parent')

from pathlib import Path

from evolution.initialization_utils import get_simple_net
from models.evaluate_model import select_action_without_random
from models.caching_environment_maker import GVGAI_RUBEN, GVGAI_BAM4D

from batch_data_prep.file_rw_utils import parse_name

def attack_to_win(act, record):
    return int(act) == 5


def attack_to_lose(act, record):
    return int(act) != 5


def match_move(act, record):
    return int(act) == int(record)



def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


datasets = [
    {
        'dir': '*keyget_*.npy',
        'func': match_move,
    },
    {
        'dir': '*winseq_*.npy',
        'func': match_move,
    },
    # {
    #     'dir': '*attL*.npy',
    #     'func': attack_to_lose,
    # },
    {
        'dir': '*attW*.npy',
        'func': attack_to_win,
    }
]


#saves_numpy = Path(__file__).parent.parent / 'saves_numpy'
saves_numpy = Path('/scratch/bw1879/quality-diversity-rl/saves_numpy/')

print('saves path', str(saves_numpy))
gvgai_version = GVGAI_BAM4D
game = 'gvgai-zelda'




policy_net, init_model = get_simple_net()
policy_net.__init__(*init_model)


num_epochs = 500
minibatch = 50

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(policy_net.parameters(), lr=0.001, momentum=0.9)

all_screens = []
all_labels = []


for epoch in range(num_epochs):
    running_loss = 0
    optimizer.zero_grad()

    for data in datasets:
        dir = data['dir']
        eval_function = data['func']
        i = 0
        cume_loss = 0
        for file in saves_numpy.glob(dir):
            parts = parse_name(file.stem)
            try:
                screen = np.load(str(file))
            except Exception as e:
                logging.info('unable to load {}'.format(str(e)))
            # output = eval_function(model_action, parts['act'])
            # all_screens.append(torch.tensor(screen))
            act = int(parts['act'])
            if act != -10:
                all_screens.append(screen[0])
                all_labels.append(act)
            i += 1

            if i % minibatch == 0:

                screens_f = torch.tensor(all_screens)
                model_actions = policy_net(screens_f)

                # out_f = one_hot_embedding(all_outputs, 10)
                label_f = torch.tensor(all_labels)
                # print('shapes', model_actions.shape, label_f.shape)
                loss = criterion(model_actions, label_f)
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                cume_loss += loss_val
                logging.info('[{} {}] loss: {}'.format(epoch + 1, i, loss_val))

                all_screens.clear()
                all_labels.clear()


        if len(all_screens):
            screens_f = torch.tensor(all_screens)
            model_actions = policy_net(screens_f)

            # out_f = one_hot_embedding(all_outputs, 10)
            label_f = torch.tensor(all_labels)
            # print('shapes', model_actions.shape, label_f.shape)
            loss = criterion(model_actions, label_f)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            cume_loss += loss_val
            logging.info('[{}] loss: {}'.format(epoch + 1, i, cume_loss))

            all_screens.clear()
            all_labels.clear()

        logging.info('{} for dir {}'.format(i, dir))
        logging.info('cume loss: {} \n\n\n\n'.format(cume_loss))
        cume_loss = 0


PATH = './policy_net.pth'
torch.save(policy_net.state_dict(), PATH)

print('hi')
