import torch.nn as nn
import torch.nn.functional as F



class a2c(nn.Module):

    def __init__(self, h, w, linear_input_scalar, kernel_size, outputs):
        super(a2c, self).__init__()

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

        linear_input_size = convw * convh * linear_input_scalar
        #self.fc1 = nn.Linear(linear_input_size,2*linear_input_size)
        #self.fc2 = nn.Linear(2*linear_input_size, outputs)
        self.actor = nn.Linear(linear_input_size, outputs)
        self.critic = nn.Linear(linear_input_size, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        action_prob = F.softmax(self.actor(x.view(x.size(0), -1)),dim =-1)
        state_values = self.critic(x.view(x.size(0), -1))

        return action_prob, state_values
