import torchvision.models as models
import torch
from torch import nn

class model_feature(nn.Module):
   def __init__(self, hidden_dim):
      super(model_feature, self).__init__()
      self.linear1 = nn.Linear(2 * 14 * 14, hidden_dim)
      self.linear2 = nn.Linear(hidden_dim, hidden_dim)
      self.relu = nn.ReLU(True)

   def logits(self, input):
      x = self.linear1(input)
      x = self.relu(x)
      x = self.linear2(x)
      x = self.relu(x)
      return x

   def forward(self, input):
      input = input.view(input.shape[0], 2 * 14 * 14).cuda()
      x = self.logits(input)
      return x

class model_task(nn.Module):
   def __init__(self, model_feature, hidden_dim, num_classes):
      super(model_task, self).__init__()
      self.num_classes = num_classes
      self.linear1 = nn.Linear(hidden_dim, num_classes)
      self.model_feature = model_feature
      self.relu = nn.ReLU(True)

   def logits(self, input):
      x = self.model_feature(input)
      x = self.linear1(x)
      return x

   def forward(self, input):
      x = self.logits(input)
      # x = torch.squeeze(x)
      return x

class model_regularizer(nn.Module):
   def __init__(self, hidden_dim, num_classes):
      super(model_regularizer, self).__init__()
      self.num_classes = num_classes
      self.linear1 = nn.Linear(hidden_dim * num_classes, 1, bias=False)

   def logits(self, input):
      x = self.linear1(input)
      return x

   def forward(self, input):
      input = input.cuda()
      x = self.logits(input)
      return x