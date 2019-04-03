import torch.nn as nn
import torch.nn.functional as F
from blocks import ResidualBlock
import torch


class Generator(nn.Module):
    def __init__(self, conv_channels,trans_channels, kernels, strides, n_residual_blocks=9, leaky_slope=0.2):
        super(Generator, self).__init__()

        models=[nn.Dropout(p=0.8,inplace=True)]
        for i in range(len(conv_channels)-1):
            models+=[
                nn.Conv1d(conv_channels[i],conv_channels[i+1],kernel_size=kernels[i], stride=strides[i], padding=1),
                # nn.Dropout(p=0.5,inplace=True),
                nn.InstanceNorm1d(conv_channels[i+1]),
                nn.ReLU(inplace=True)
            ]
            
        
        for _ in range(n_residual_blocks):
            models+=[
                ResidualBlock(conv_channels[-1],leaky_slope=leaky_slope)
            ]
        
        kernels=kernels[len(conv_channels):]
        strides=strides[len(conv_channels):]
        for i in range(len(trans_channels)-1):
            models+=[
                nn.ConvTranspose1d(trans_channels[i],trans_channels[i+1],kernel_size=kernels[i], stride=strides[i], padding=1, output_padding=1),
                # nn.Dropout(p=0.5, inplace=True),
                nn.InstanceNorm1d(trans_channels[i+1]),
                nn.ReLU(inplace=True)
            ]

        models+=[
            nn.Conv1d(trans_channels[-1],conv_channels[0], kernel_size=kernels[-1], stride=strides[-1], padding=1),
            nn.InstanceNorm1d(conv_channels[0]),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*models)


        # # downsampling
        # self.c1 = nn.Conv1d(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        # self.c2 = nn.Conv1d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
        # self.c3 = nn.Conv1d(channels[2], channels[3], kernel_size=3, stride=2, padding=1)


        # # res blocks
        # self.res_blocks= nn.ModuleList()
        # for _ in range(n_residual_blocks):
        #     self.res_blocks.append(ResidualBlock(128,))

        # # upsampling
        # self.tc1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.tc2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.c4 = nn.Conv1d(32,nc_output, kernel_size=3,stride=1,padding=1)
        
        # self.leaky_relu=nn.LeakyReLU(negative_slope=0.2)


    def forward(self, x):
        return self.model(x)