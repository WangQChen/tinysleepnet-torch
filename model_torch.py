# model.py (PyTorch version of TinySleepNet)

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics


class TinySleepNet(nn.Module):
    def __init__(self, config):
        super(TinySleepNet, self).__init__()
        self.config = config
        self.use_rnn = config.get("use_rnn", False)
        self.seq_length = config.get("seq_length", 20)

        # CNN layers
        self.conv1 = nn.Conv1d(1, 128, kernel_size=config['sampling_rate'] // 2, stride=config['sampling_rate'] // 16)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(8)

        self.conv2_1 = nn.Conv1d(128, 128, kernel_size=8)
        self.bn2_1 = nn.BatchNorm1d(128)
        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=8)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.conv2_3 = nn.Conv1d(128, 128, kernel_size=8)
        self.bn2_3 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)

        # compute flatten size
        dummy_input = torch.zeros(1, 1, config['input_size'])
        out = self._forward_features(dummy_input)
        self.feature_size = out.shape[1]

        self.fc = nn.Linear(self.feature_size, 512)
        self.out = nn.Linear(512, config['n_classes'])

        if self.use_rnn:
            self.rnn = nn.LSTM(512, config['n_rnn_units'], config['n_rnn_layers'], batch_first=True)
            self.out = nn.Linear(config['n_rnn_units'], config['n_classes'])

    def _forward_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout(x, 0.5, training=self.training)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn2_3(self.conv2_3(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, 0.5, training=self.training)
        return x

    def forward(self, x, seq_lengths=None):
        x = self._forward_features(x)
        x = F.relu(self.fc(x))
        if self.use_rnn:
            B = x.shape[0] // self.seq_length
            x = x.view(B, self.seq_length, -1)
            x, _ = self.rnn(x)
            x = x.contiguous().view(B * self.seq_length, -1)
        return self.out(x)