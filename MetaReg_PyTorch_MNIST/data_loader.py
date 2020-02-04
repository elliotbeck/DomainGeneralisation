import torch.utils.data as data
import util
import numpy as np
import torch
import h5py

class mnist(data.Dataset):
    def __init__(self, data, e):
        super(mnist, self).__init__()
        self.data = data[0]
        self.target = data[1]
        self.e = e

    def __getitem__(self, index):
        return util.preprocess(self.data[index,:,:], self.target[index], self.e)
        
    def __len__(self):
        return self.data.shape[0]