import numpy as np
from pathlib import Path
from shutil import copyfile
from datetime import datetime
from collections import defaultdict
from batch_data_prep.data_drive import drive, out_dir


is_mac = True
do_hash = True

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

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

out_path = Path(out_dir(is_mac=is_mac))
if not do_hash:
    out_path = out_path / 'with_dups'

paths = dict()
for category in ['win', 'lose', 'other']:
    paths[category] = out_path / category

if do_hash:
    start = datetime.now()
    files = Path(drive(is_mac=is_mac)).glob('*.npy')
    print('total', len(list(files)))
    imgs = defaultdict(list)
    for i, file_path in enumerate(Path(drive(is_mac=is_mac)).glob('*.npy')):
        img = np.load(open(str(file_path), 'rb'))
        end = datetime.now()
        imgs[hash(img.tostring())].append({'name': file_path.stem, 'path': file_path})
    # imgs = {k: v for k, v in sorted(imgs.items(), key=lambda item: len(item[1]), reverse=True)}
    end = datetime.now()
    print('loaded hashes in', end - start)


    for hash_, file_list in imgs.items():
        first_file = file_list[0]
        labels = parse_name(first_file['name'])
        reward = int(float(labels['reward']))
        if reward not in [0, -10]:
            act_cat = labels['act'] if labels['act'] == '1' else 'other'
            specific_out_path = paths[labels['crit']] / labels['reward'] / act_cat
            if not specific_out_path.exists():
                specific_out_path.mkdir(parents=True)
            copyfile(str(first_file['path']), str(specific_out_path / first_file['path'].stem) + '.npy')

else:
    for file_path in Path(drive(is_mac=is_mac)).glob('*.npy'):
        first_file = {'name': file_path.stem, 'path': file_path}
        labels = parse_name(first_file['name'])
        reward = int(float(labels['reward']))
        if reward not in [0, -10]:
            act_cat = labels['act'] if labels['act'] == '1' else 'other'
            specific_out_path = paths[labels['crit']] / labels['reward'] / act_cat
            if not specific_out_path.exists():
                specific_out_path.mkdir(parents=True)
            copyfile(str(first_file['path']), str(specific_out_path / first_file['path'].stem) + '.npy')
