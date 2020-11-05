"""# Model

"""

import random

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms



class NET(nn.Module):

    def __init__(self):
        super(NET, self).__init__()

        # conv1
        self.conv1_1 = nn.Conv2d(1, 16, 3, padding=1) # H * w * 16
        self.relu = nn.ReLU()
        self.conv1_2 = nn.Conv2d(16, 32, 3, padding=1) # H * w * 32
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H/2 * W/2 * 32

        # conv2
        self.conv2_1 = nn.Conv2d(32, 32, 3, padding=1) # H / 2 * w/2 * 32
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1) # H / 2 * w/2 * 32
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H /4 * W / 4 * 32

        # conv3
        self.conv3_1 = nn.Conv2d(32, 64, 3, padding=1) # H / 4 * w/4 * 64
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=1) # H / 4 * w/4 * 64
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H /8 * W / 8 * 64

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) # H / 4 * H /4 * 32
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) # H / 2 * H /2 * 16
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) # H  * H * 8
        self.conv4 = nn.Conv2d(8, 1, 3, padding=1) # H *V
        self.sigmoid = nn.Sigmoid()



        # conv3
        """ self.fc3 = nn.Conv2d(32, 16,  7,padding= 3)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout2d()

        self.fc4 = nn.Conv2d(32, 16, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(16, 16, 1)
        self.upscore = nn.ConvTranspose2d(16, 1, 64, stride=4,bias=False)
        """
        #end


    def forward(self, x):
        h = self.relu(self.conv1_1(x))
        h = self.relu(self.conv1_2(h))
        h = self.pool1(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.pool2(h)
        h = self.relu(self.conv3_1(h))
        h = self.pool3(h)

        h = self.relu(self.deconv1(h))
        h = self.relu(self.deconv2(h))
        h = self.relu(self.deconv3(h))
        h = self.sigmoid(self.conv4(h))
        return h



class FullNET(nn.Module):

    def __init__(self):
        super(FullNET, self).__init__()

        # conv1
        self.conv1_1 = nn.Conv2d(1, 16, 3, padding=1) # H * w * 16
        self.relu = nn.ReLU()
        self.conv1_2 = nn.Conv2d(16, 32, 3, padding=1) # H * w * 32
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H/2 * W/2 * 32

        # conv2
        self.conv2_1 = nn.Conv2d(32, 32, 3, padding=1) # H / 2 * w/2 * 32
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1) # H / 2 * w/2 * 32
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H /4 * W / 4 * 32

        # conv3
        self.conv3_1 = nn.Conv2d(32, 64, 3, padding=1) # H / 4 * w/4 * 64
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=1) # H / 4 * w/4 * 64
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H /8 * W / 8 * 64

        self.deconv1 = nn.Conv2d(64, 32, 3, padding=1) # H / 4 * H /4 * 32
        self.deconv2 = nn.Conv2d(32, 16, 3, padding=1) # H / 2 * H /2 * 16
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H /8 * W / 8 * 64

        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) # H / 4 * H /4 * 32
        self.deconv4 = nn.Conv2d(8, 4, 3, padding=1) # H  * H * 8
        self.pool5 = nn.MaxPool2d(2, stride=2 ,ceil_mode=True)  # H /8 * W / 8 * 64

        self.conv4 = nn.ConvTranspose2d(4, 2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) # H / 4 * H /4 * 32
        self.conv5 = nn.Conv2d(2, 2, 1, padding=0) # H *V



        # conv3
        """ self.fc3 = nn.Conv2d(32, 16,  7,padding= 3)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout2d()

        self.fc4 = nn.Conv2d(32, 16, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(16, 16, 1)
        self.upscore = nn.ConvTranspose2d(16, 1, 64, stride=4,bias=False)
        """
        #end


    def forward(self, x):
        h = self.relu(self.conv1_1(x))
        h = self.relu(self.conv1_2(h))
        h = self.pool1(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.pool2(h)
        h = self.relu(self.conv3_1(h))
        h = self.pool3(h)

        h = self.relu(self.deconv1(h))
        h = self.relu(self.deconv2(h))
        h = self.pool4(h)
        h = self.relu(self.deconv3(h))
        h = self.relu(self.deconv4(h))
        h = self.pool5(h)
        h = self.relu(self.conv4(h))
        h = self.relu(self.conv5(h))
        return h
