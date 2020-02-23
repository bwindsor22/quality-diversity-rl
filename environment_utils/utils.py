import torch
from datetime import datetime

# find time once at file load
time = datetime.now().strftime("%Y-%m-%d-%H-%M-%s")

def find_device():
    return torch.device("cpu")


def get_run_file_name():
    return '{}-runfile.log'.format(time)

def get_run_name():
    return time
