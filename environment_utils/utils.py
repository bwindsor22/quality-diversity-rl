import torch
from datetime import datetime

# find time once at file load
time = datetime.now().strftime("%Y-%m-%d-%H-%M-%s")

def find_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_run_file_name():
    return '{}-runfile.log'.format(time)