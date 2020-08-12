import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from pathlib import Path

from evolution.initialization_utils import get_initial_model
from models.evaluate_model import select_action_without_random
from models.caching_environment_maker import GVGAI_RUBEN

data = []

labels = [True] * len(data)


def attack_to_score(act, record):
    return int(act) == 1


def attack_to_lose(act, record):
    return int(act) != 1


def move_to_reward1(act, record):
    return int(act) == int(record)


def move_to_win(act, record):
    return int(act) == int(record)

def parse_name(file_name):
    items = dict()
    parts = file_name.split('_')
    items['uuid'] = parts[0]
    items['lvl'] = parts[1]
    is_name = True
    name = ''
    for part in parts[2:]:
        if is_name:
            name = part
            is_name = False
        else:
            items[name] = part
            is_name = True
    return items


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
        'dir': '*act_1_reward_2.0_crit_other*.npy',
        'func': attack_to_score,
    },
    # {
    #     'dir': '*act_1_*crit_lose*.npy',
    #     'func': attack_to_lose,
    # },
    {
        'dir': '*reward_1.0_crit_other*.npy',
        'func': move_to_reward1,
    },
    {
        'dir': '*reward_1.0_crit_win*.npy',
        'func': move_to_win,
    },
]


saves_numpy = Path(__file__).parent.parent / 'saves_numpy'
gvgai_version = GVGAI_RUBEN
game = 'gvgai-zelda'




policy_net, init_model = get_initial_model(gvgai_version, game)
policy_net.__init__(*init_model)


num_epochs = 50

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(policy_net.parameters(), lr=0.001, momentum=0.9)

all_outputs = []
all_screens = []
all_labels = []

for data in datasets:
    dir = data['dir']
    eval_function = data['func']
    i = 0
    for file in saves_numpy.glob(dir):
        parts = parse_name(file.stem)
        screen = np.load(str(file))
        # output = eval_function(model_action, parts['act'])
        # all_screens.append(torch.tensor(screen))
        all_screens.append(screen[0])
        all_labels.append(int(parts['act']))
        i += 1
    print(i, 'for dir', dir)

for epoch in range(num_epochs):
    running_loss = 0
    optimizer.zero_grad()

    screens_f = torch.tensor(all_screens)
    model_actions = policy_net(screens_f)

    # out_f = one_hot_embedding(all_outputs, 10)
    label_f = torch.tensor(all_labels)
    print('shapes', model_actions.shape, label_f.shape)
    loss = criterion(model_actions, label_f)
    loss.backward()
    optimizer.step()
    print('[{}] loss: {}'.format(epoch + 1, loss.item()))
    # all_screens.clear()
    # all_outputs.clear()
    # all_labels.clear()

PATH = './policy_net.pth'
torch.save(policy_net.state_dict(), PATH)

print('hi')
