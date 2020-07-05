from pprint import pprint
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
saves_dir = '/Users/bradwindsor/ms_projects/qd-gen/gameQD/saves'
all_files = list(Path(saves_dir).glob('*.pkl'))
print(len(all_files))
counter = defaultdict(int)



saves_numpy = Path(__file__).parent.parent / 'saves_numpy'
for file_ in saves_numpy.glob('*'):
    file_.unlink()
saves_numpy.mkdir(exist_ok=True)

start = datetime.now()
count = 0
for file_ in all_files:
    dps = pickle.load(open(str(file_), 'rb'))
    for dp in dps:
        # point = dp['type'] + '_' + str(int(dp['reward']))
        crit = dp['critical']
        rew = int(dp['reward'])
        point = crit + '_' + str(rew)
        counter[point] += 1
        if point == 'no_0':
            #skip: random screen saved for evaluator
            continue
        action = dp['action']
        screen = dp['current_screen']
        out_file = saves_numpy / f'c_{count}_crit_{crit}_rew_{rew}_act_{action}.npy'
        np.save(str(out_file), screen)
        count += 1

print('processed all files in ', datetime.now() - start)
c2 = dict(counter)
pprint(c2)

start = datetime.now()
for file in saves_numpy.glob('*.npy'):
    a = np.load(open(str(file), 'rb'))

print('reloaded all files in ', datetime.now() - start)
