import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

parser = argparse.ArgumentParser(description='PACS')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs_metatrain', type=int, default=10)
parser.add_argument('--epochs_full', type=int, default=15)
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--meta_train_steps', type=int, default=20)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()

# Load MNIST, make train/val splits, and shuffle train set examples

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_train = (mnist.train_data, mnist.train_labels)
mnist = datasets.MNIST(root='./data', train=False, download=True, transform=None)
mnist_val = (mnist.test_data, mnist.test_labels)

def make_environment(images, labels, e):
    def torch_bernoulli(p, size):
      return (torch.rand(size) < p).float()
    def torch_xor(a, b):
      return (a-b).abs() # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.),
      'labels': labels[:, None]
    }

envs = [
    make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
    make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
    make_environment(mnist_val[0], mnist_val[1], 0.9)
  ]

  
class MLP(nn.Module):
    def __init__(self):
      super(MLP, self).__init__()
      if flags.grayscale_model:
        lin1 = nn.Linear(14 * 14, flags.hidden_dim)
      else:
        lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
      lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
      lin3 = nn.Linear(flags.hidden_dim, 1)
      for lin in [lin1, lin2, lin3]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
      self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    def forward(self, input):
      if flags.grayscale_model:
        out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
      else:
        out = input.view(input.shape[0], 2 * 14 * 14)
      out = self._main(out)
      return out

mlp = MLP()

def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

optimizer = optim.Adam(mlp.parameters(), lr=0.001)

for step in range(10):
    loss1 = mean_nll(mlp(envs[0]['images']),envs[0]['labels'])
    loss2 = mean_nll(mlp(envs[1]['images']),envs[1]['labels'])
    loss = loss1 + loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(mean_accuracy(mlp(envs[0]['images']),envs[0]['labels']))
    print(mean_accuracy(mlp(envs[2]['images']),envs[2]['labels']))


#### My model ####
from data_loader import mnist
from torch.utils import data
# load data
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=None)
# 30000 obs. per train set
train_data1 = (mnist_train.train_data[::2], mnist_train.train_labels[::2])
train_data2 = (mnist_train.train_data[1::2], mnist_train.train_labels[1::2]) 
# 10000 obs. in test set
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=None)
test_data = (mnist_test.test_data, mnist_test.test_labels)

# put data in dataloader
train_data1 = mnist(train_data1, 0.2)
train_data2 = mnist(train_data2, 0.1)
test_data = mnist(test_data, 0.9)
train_data_full = data.DataLoader(data.ConcatDataset([train_data1, train_data2]), num_workers=1, batch_size=flags.batch_size, 
                              shuffle=True, drop_last=True)
train_data1 = data.DataLoader(train_data1, num_workers=1, batch_size=flags.batch_size, 
                              shuffle=True, drop_last=True)
train_data2 = data.DataLoader(train_data2, num_workers=1, batch_size=flags.batch_size, 
                              shuffle=True, drop_last=True)                                
test_data = data.DataLoader(test_data, num_workers=1, batch_size=flags.batch_size, 
                              shuffle=True, drop_last=True)    

def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

class MLP(nn.Module):
    def __init__(self):
      super(MLP, self).__init__()
      if flags.grayscale_model:
        lin1 = nn.Linear(14 * 14, flags.hidden_dim)
      else:
        lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
      lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
      lin3 = nn.Linear(flags.hidden_dim, 1)
      for lin in [lin1, lin2, lin3]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
      self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    def forward(self, input):
      if flags.grayscale_model:
        out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
      else:
        out = input.view(input.shape[0], 2 * 14 * 14)
      out = self._main(out)
      return out

mlp = MLP()

optimizer = optim.Adam(mlp.parameters(), lr=0.001)


for step in range(10):
  accuracy_train = 0
  accuracy_test = 0
  for i, batch in enumerate(train_data1):
    loss1 = mean_nll(torch.squeeze(mlp(batch[0])),batch[1])
    loss = loss1 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    accuracy_batch = mean_accuracy(torch.squeeze(mlp(batch[0])),batch[1])
    accuracy_train += accuracy_batch
  print(accuracy_train/i)

  for i, batch in enumerate(test_data):
    accuracy_batch = mean_accuracy(torch.squeeze(mlp(batch[0])),batch[1])
    accuracy_test += accuracy_batch
  print(accuracy_test/i)