import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityShortcut(nn.Module):
    """
        A class to make Identity Mapping Shortcut

        Attributes:
            pooling : A max pooling layer 
            extra_cahnnel : The difference between input an output channels
    """
    def __init__(self, in_channels, out_channels, stride):
        """
            Initialize the Identity Shortcut Class

            Args:
                in_channels: number of input channels
                out_channels: number of output channels
                stride: size of stride

            Returns:
                None
        """
        super().__init__()
        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.extra_channel = out_channels - in_channels
    
    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 0, self.extra_channel))
        x = self.pooling(x)
        return x

class ResidualBlock(nn.Module):
    """
        A class about residual block

        When stride == 1, just add input value to output of residual block
        When stride == 2, process the input value according to shortcut type and add it to output of residual block
        Attributes:
            residual_blcok: A sequential container to sequentially configure the layers for residual learning
            shortcut: A shortcut connection of residual block
            relu: A relu activation layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, shortcut_type = None):
        """
            Initialize the residual block

            Args:
                in_channels: The number of input channels
                out_channels: The number of output channels
                kernel_size: The number of kernnel size in the convolution layers
                stride: The size of stride
                shortcut_type: The type of shortcut connection when stride is not 1
            Returns:
                None
        """
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        
        if stride != 1:
            if shortcut_type == 'A':
                self.shortcut = IdentityShortcut(in_channels, out_channels, stride)
            elif shortcut_type == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride),
                    nn.BatchNorm2d(out_channels)
                )
        
    
    def forward(self, x):
        x = self.residual_block(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    """
        A class about residual network

        Attributes:
            conv1_x: First set of layers
            conv2_x: Second set of residual blocks
            conv3_x: Third set of residual blocks
            conv4_x: Fourth set of residual blocks
            avg_pool: A average pooling layer of residual network
            flatten: A flatten layer for the fully connected layer
            fc: A fully connected layer to get probability of each class about given image
    """
    def __init__(self, conv_num, num_classes):
        """
            Initialize the residual network

            Args:
                conv_num: The number of residual blocks in each set of residual block
                num_classes: The number of classes to predict
            Returns:
                None
        """
        super().__init__()
        self.conv1_x = nn.Sequential(
                                    nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding=1, bias=False),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    #nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        )
        
        self.conv2_x = nn.Sequential(*[ResidualBlock(16, 16, 3, 1) for _ in range(conv_num[0])])
        self.conv3_x = nn.Sequential(ResidualBlock(16, 32, 3, 2, 'A'), *[ResidualBlock(32, 32, 3, 1) for _ in range(conv_num[1] - 1)])
        self.conv4_x = nn.Sequential(ResidualBlock(32, 64, 3, 2, 'A'), *[ResidualBlock(64, 64, 3, 1) for _ in range(conv_num[2] - 1)])
        #self.conv5_x = nn.Sequential(ResidualBlock(256, 128, 3, 2), *[ResidualBlock(512, 512, 3, 1) for _ in range(conv_num[3])])

        #self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64,out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        #x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x