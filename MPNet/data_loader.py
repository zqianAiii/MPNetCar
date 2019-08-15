#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import os
import pickle
import torch
import torch.nn as nn
import torch.utils.data as td


# In[ ]:


def read_path(filename):
       
    f = open(filename, 'r')
    
    obs = []
    while True:
        line = f.readline()
        if line != '\n':
            obs.append(tuple(float(i) for i in line.split()))
        else:
            break
            
    dataset = []
    occmap = []
    path = []
    while True:
        line = f.readline()
        if line != '' and line != '\n':
            temp = list(float(i) for i in line.split())
            if len(temp) == 3:
                path.append([temp[0]/20, temp[1]/20, temp[2]/math.pi])
            else:
                path.append(temp)
        elif line == '\n':
            pc = np.zeros((256, 256), dtype = np.float32)
            path_obs = create_occmap(obs)
            for x, y in path_obs:
                pc[255-y, x] = 1
            for i, config in enumerate(path[:-2]):
                dataset.append(config + path[-2] + path[i+1])
                occmap.append(pc)
            path = []
        else:
            break

    f.close()
    return occmap, dataset


# In[ ]:


def create_occmap(obs, num_pt = 2000):
    pc = []
    num = 0
    num_pt = np.random.randint(num_pt, high = 2*num_pt)
    while num < num_pt:
        x = np.random.randint(256)
        y = np.random.randint(256)
        for i in obs:
            if in_rectangle(x, y, i):
                pc.append((x, y))
                num += 1
                break
    return pc


# In[ ]:


def in_rectangle(x, y, rect, ratio = 256/20):
    x_p = (x - ratio*rect[0]) * np.cos(-ratio*rect[2]) - (y - ratio*rect[1]) * np.sin(-ratio*rect[2])
    y_p = (x - ratio*rect[0]) * np.sin(-ratio*rect[2]) + (y - ratio*rect[1]) * np.cos(-ratio*rect[2])
    if abs(x_p) <= (ratio*rect[4]/2) and abs(y_p) <= (ratio*rect[3]/2):
        return True
    else:
        return False


# In[ ]:


class PathDataset(td.Dataset):
    
    def __init__(self, filenames, image_size = (256, 256)):
        super(PathDataset, self).__init__()
        self.image_size = image_size
        self.pc, self.config = self.import_path(filenames)
        
    def import_path(self, filenames):
        config = []
        pc = []
        for name in filenames:
            occmap, data = read_path(name)
            pc += occmap
            config += data
            print('Finished reading %s' % name)
        return pc, config
        
    def __len__(self):
        return len(self.config)
    
    def __getitem__(self, idx):
        pc = torch.Tensor(self.pc[idx]).view(1, self.image_size[0], self.image_size[1])
        x = torch.Tensor(self.config[idx][:6])
        y = torch.Tensor(self.config[idx][6:])
        return pc, x, y
    


# In[ ]:


def get_loader(filenames, batch_size = 1024, validation_split = 0.1):
    
    try:
        with open('dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
    except:
        dataset = PathDataset(filenames)
        with open('dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
    
    indices = list(range(len(dataset)))
    split = int(np.floor(validation_split * len(dataset)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = td.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = td.sampler.SubsetRandomSampler(val_indices)
    
    train_loader = td.DataLoader(dataset, batch_size = batch_size, sampler = train_sampler, pin_memory = True, num_workers = 0)
    val_loader = td.DataLoader(dataset, batch_size = batch_size, sampler = valid_sampler, pin_memory = True, num_workers = 0)
    
    return train_loader, val_loader


# In[ ]:




