def drive(is_mac=True):
    if is_mac:
        return '/Users/bradwindsor/ms_projects/qd-gen/gameQD/saves_numpy/'
    else:
        return '/scratch/bw1879/quality-diversity-rl/saves_numpy/'

def out_dir(is_mac=True):
    if is_mac:
        return '/Users/bradwindsor/ms_projects/qd-gen/gameQD/saves_formatted/'
    else:
        return '/scratch/bw1879/quality-diversity-rl/saves_formatted/'
