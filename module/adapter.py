from typing import Any
import torch
import torch.nn as nn
import numpy as np


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

class Linear(nn.Module):
    def __init__(self, hidden_size=256, output_size=768):
        super(Linear, self).__init__()
        self.adapter = nn.Linear(hidden_size, output_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda()
        self.logit_scale_v2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda()
        conv_init(self.adapter)
    
    def forward(self, x):
        return self.adapter(x)
    
    def get_logit_scale(self):
        return self.logit_scale
    
    def get_logit_scale_v2(self):
        return self.logit_scale_v2




class Adapter(nn.Module):
    def __init__(self, hidden_size=256, output_size=768):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, output_size, bias=False)
        self.act = nn.GELU()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda()
        conv_init(self.fc1)
        conv_init(self.fc2)
        conv_init(self.fc3)
    
    def forward(self, x):
        xs = self.fc1(x)
        xs = self.act(xs)
        xs = self.fc2(x)
        return self.fc3(x) + xs
    
    def get_logit_scale(self):
        return self.logit_scale


