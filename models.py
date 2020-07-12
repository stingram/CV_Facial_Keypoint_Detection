## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import numpy as np

class Net(nn.Module):

    def __init__(self, out_size):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        input_shape = (1,224,224)
        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2),
                                  nn.Conv2d(32,64, kernel_size=5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2),
                                  nn.Conv2d(64,128, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2),
                                  nn.Conv2d(128,256, kernel_size=3),
                                  nn.ReLU())
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512),
                                nn.ReLU(),
                                nn.Dropout(p=0.25),
                                nn.Linear(512,256),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(256, 136))
        
        
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        conv_out = self.conv(x).view(x.size()[0], -1)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return self.fc(conv_out)
