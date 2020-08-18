import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, h, w, linear_input_scalar, kernel_size, outputs):
        super(DQN, self).__init__()
        
        self.atari = True
        
        if self.atari == False:
        
            half_scalar = int(linear_input_scalar / 2)
            #self.conv1 = nn.Conv2d(3, half_scalar, kernel_size=kernel_size, stride=2)
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

            self.head = nn.Linear(linear_input_size, outputs)
        
        else:
            half_scalar = int(linear_input_scalar / 2)
            #self.conv1 = nn.Conv2d(3, half_scalar, kernel_size=kernel_size, stride=1)
            self.conv1 = nn.Conv2d(3, half_scalar, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(half_scalar, linear_input_scalar, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(linear_input_scalar, linear_input_scalar, kernel_size=3, stride=1)


            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
            def conv2d_size_out(size,kernel_sz, stride=1):
                return (size - (kernel_sz - 1) - 1) // stride + 1

            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)

            linear_input_size = convw * convh * linear_input_scalar
            self.fc = nn.Linear(linear_input_size, 512)   
            self.head = nn.Linear(512, outputs)        

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        
        if self.atari == False:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return self.head(x.view(x.size(0), -1))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc(x.view(x.size(0), -1)))
            return self.head(x.view(x.size(0), -1)) 
