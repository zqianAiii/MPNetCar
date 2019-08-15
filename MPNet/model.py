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
        
        self.conv = nn.Sequential(nn.Conv2d(1, 4, (7, 7), bias = False),
                                                 nn.BatchNorm2d(4),
                                                 nn.PReLU(),
                                                 nn.MaxPool2d((2, 2)),
                                                 nn.Conv2d(4, 4, (7, 7), bias = False),
                                                 nn.BatchNorm2d(4),
                                                 nn.PReLU(),
                                                 nn.MaxPool2d((2, 2)))
        
        
        self.linear = nn.Sequential(nn.Linear(13924, 2800),
                                             nn.PReLU(),
                                             nn.Linear(2800, 512),
                                             nn.PReLU(),
                                             nn.Linear(512, 256),
                                             nn.PReLU(),
                                             nn.Linear(256, 128),
                                             nn.PReLU(),
                                             nn.Linear(128, 28))
        
        nn.init.kaiming_normal_(self.state_dict()['conv.0.weight'])
        nn.init.kaiming_normal_(self.state_dict()['conv.4.weight'])
        nn.init.kaiming_normal_(self.state_dict()['linear.0.weight'])
        nn.init.kaiming_normal_(self.state_dict()['linear.2.weight'])
        nn.init.kaiming_normal_(self.state_dict()['linear.4.weight'])
        nn.init.kaiming_normal_(self.state_dict()['linear.6.weight'])
        nn.init.kaiming_normal_(self.state_dict()['linear.8.weight'])
        
    def forward(self, x):
        h0 = self.conv(x)
        y = self.linear(h0.view(-1, 13924))
        return y


# In[3]:


class Pnet(nn.Module):
    
    def __init__(self, input_size = 34, output_size = 3):
        super(Pnet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, 1280),
                                                nn.Linear(1280, 1024),
                                                nn.Linear(1024, 896),
                                                nn.Linear(896, 768),
                                                nn.Linear(768, 512),
                                                nn.Linear(512, 384),
                                                nn.Linear(384, 256),
                                                nn.Linear(256, 256),
                                                nn.Linear(256, 128),
                                                nn.Linear(128, 64),
                                                nn.Linear(64, 32),
                                                nn.Linear(32, output_size)])
        self.prelu = nn.ModuleList()
        self.dropout = nn.Dropout()
        for i in range(12):
            nn.init.kaiming_normal_(self.linears[i].weight.data)
            if i < 11:
                self.prelu.append(nn.PReLU())
        self.MSE = nn.MSELoss()
                
    def forward(self, z, config):
        x = torch.cat((z, config), 1)
        for i in range(10):
            x = self.dropout(self.prelu[i](self.linears[i](x)))
        h = self.prelu[10](self.linears[10](x))
        y = self.linears[11](h)
        return y
    
    def criterion(self, c_in, c_out):
        return self.MSE(c_out, c_in)

