import torch.nn as nn
import torch.nn.functional as F
import logging

class DQN(nn.Module):
    """
    ~7K params
    """
    def __init__(self, h, w, linear_input_scalar, kernel_size, outputs):
        super(DQN, self).__init__()
        # 90 130 8
        half_scalar = int(linear_input_scalar / 2)
        self.conv1 = nn.Conv2d(3, half_scalar, kernel_size=kernel_size, stride=2)
        self.conv2 = nn.Conv2d(half_scalar, linear_input_scalar, kernel_size=kernel_size, stride=2)
        self.conv3 = nn.Conv2d(linear_input_scalar, linear_input_scalar, kernel_size=kernel_size, stride=2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        # 832
        linear_input_size = convw * convh * linear_input_scalar

        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))


class BigNet(nn.Module):
    """
    ~90K params
    """
    def __init__(self, h, w, not_used, not_used2, outputs):
        super(BigNet, self).__init__()
        kernel_size = 5
        self.conv1 = nn.Conv2d(3, 4, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 2, kernel_size)
        self.fc1 = nn.Linear(1102, 84)
        # self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1102)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 94008 trainable params
# net = Net(1, 2, 0, 0, 10)
# print('out', sum(p.numel() for p in net.parameters()))