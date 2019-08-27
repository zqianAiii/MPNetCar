#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import torch
from torch import nn


# In[2]:


class Enet(nn.Module):
    
    def __init__(self):
        super(Enet, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(1, 4, (7, 7)),
                                            nn.PReLU(),
                                            nn.MaxPool2d((2, 2)),
                                            nn.Conv2d(4, 4, (6, 6)),
                                            nn.PReLU(),
                                            nn.MaxPool2d((2, 2)))
        
        
        self.linear = nn.Sequential(nn.Linear(14400, 512),
                                             nn.PReLU(),
                                             nn.Linear(512, 28))
        
        nn.init.kaiming_normal_(self.state_dict()['conv.0.weight'])
        nn.init.kaiming_normal_(self.state_dict()['conv.3.weight'])
        nn.init.kaiming_normal_(self.state_dict()['linear.0.weight'])
        nn.init.kaiming_normal_(self.state_dict()['linear.2.weight'])
        
    def forward(self, x):
        h0 = self.conv(x)
        y = self.linear(h0.view(-1, 14400))
        return y


# In[3]:


class Pnet(nn.Module):
    
    def __init__(self, input_size = 34, output_size = 3):
        super(Pnet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, 1280),
                                                nn.Linear(1280, 896),
                                                nn.Linear(896, 512),
                                                nn.Linear(512, 384),
                                                nn.Linear(384, 256),
                                                nn.Linear(256, 128),
                                                nn.Linear(128, 64),
                                                nn.Linear(64, 32),
                                                nn.Linear(32, output_size)])
        self.prelu = nn.ModuleList()
        self.dropout = nn.Dropout()
        for i in range(9):
            nn.init.kaiming_normal_(self.linears[i].weight.data)
            if i < 8:
                self.prelu.append(nn.PReLU())
        self.MSE = nn.MSELoss()
        
    def forward(self, z, config):
        x = torch.cat((z, config), 1)
        for i in range(7):
            x = self.dropout(self.prelu[i](self.linears[i](x)))
        h = self.prelu[7](self.linears[7](x))
        y = self.linears[8](h)
        return y
    
    def criterion(self, c_in, c_out):
        return self.MSE(c_out, c_in)

