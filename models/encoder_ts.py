import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class EncoderConv(nn.Module):
    def __init__(self, input_dim, seq_len, h_dims, kernel_size = 5):
        super().__init__()
        padding = kernel_size // 2
        self.encoding_dim = h_dims[-1]
        
        assert len(h_dims) == 3 

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=h_dims[0],
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm1d(h_dims[0]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(0.35)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=h_dims[0],
                out_channels=h_dims[1],
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm1d(h_dims[1]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(0.35)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=h_dims[1],
                out_channels=h_dims[2],
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm1d(h_dims[2]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        return x


class EnconderMLP(nn.Module):
    def __init__(self, input_dim, seq_len, h_dims):
        super().__init__()
        in_dim = input_dim * seq_len
        self.encoding_dim = h_dims[-1]
        self.layers = []
        for h_dim in h_dims:
            self.layers.append(nn.Linear(in_dim, h_dim))
            self.layers.append(nn.ReLU())
            in_dim = h_dim
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


