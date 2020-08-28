import numpy as np
from itertools import islice
import torch
import torch.optim as optim
import torch.nn as nn
import logging

logging.basicConfig(filename='../logs/parent-{}.log'.format('train'), level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info('Initializing parent')

from pathlib import Path

from evolution.initialization_utils import get_simple_net
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


is_mac = True
#saves_numpy = Path(__file__).parent.parent / 'saves_numpy'
if is_mac:
    saves_numpy = Path('/Users/bradwindsor/ms_projects/qd-gen/gameQD/saves_server_sample_2/')
else:
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
loss_record = []

def get_generators():
    ### Make Generators
    generators = []
    gen_diffs = []
    gen_start = 0
    gen_diff = 1000
    while gen_start < 1000000:
        gen_diffs.append((gen_start, gen_start + gen_diff))
        gen_start += gen_diff
    for gen_start, gen_end in gen_diffs:
        for data in datasets:
            dir = data['dir']
            generators.append((islice(saves_numpy.glob(dir), gen_start, gen_end), dir))
    return generators

for epoch in range(num_epochs):
    running_loss = 0
    optimizer.zero_grad()

    cume_loss = 0
    epoch_counts = []


    ### Use Generators
    for generator, dir in get_generators():
        i = 0
        for file in generator:
            parts = parse_name(file.stem)
            try:
                screen = np.load(str(file))
            except Exception as e:
                logging.info('unable to load {}'.format(str(e)))

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
        epoch_counts.append(i)
        if len(epoch_counts) > len(datasets) and sum(epoch_counts[(-2*len(datasets)):]) == 0:
            break
    logging.info('epoch {} cume loss: {} \n\n'.format(epoch, cume_loss))
    cume_loss = 0
    loss_record.append((epoch, cume_loss))

print('all losses')
print(loss_record)

PATH = './policy_net.pth'
torch.save(policy_net.state_dict(), PATH)

print('hi')
