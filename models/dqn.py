import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, h, w, linear_input_scalar, kernel_size, outputs):
        super(DQN, self).__init__()

        half_scalar = int(linear_input_scalar / 2)
        self.conv1 = nn.Conv2d(3, half_scalar, kernel_size=kernel_size, stride=2)
        self.bn1 = nn.BatchNorm2d(half_scalar)
        self.conv2 = nn.Conv2d(half_scalar, linear_input_scalar, kernel_size=kernel_size, stride=2)
        self.bn2 = nn.BatchNorm2d(linear_input_scalar)
        self.conv3 = nn.Conv2d(linear_input_scalar, linear_input_scalar, kernel_size=kernel_size, stride=2)
        self.bn3 = nn.BatchNorm2d(linear_input_scalar)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        # Original Height: 110
        # linear_input_size = convw * convh * 32

        # V2 Height: 55
        # linear_input_size = convw * convh * 16

        # V3 Height: 28
        linear_input_size = convw * convh * linear_input_scalar

        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
