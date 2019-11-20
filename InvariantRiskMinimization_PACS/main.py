import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from torch.nn import functional as F
import torchvision.models as models
import h5py

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=2)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
  print("Restart", restart)

  # Load MNIST, make train/val splits, and shuffle train set examples


  # read in training data

  # Get the Art training data
  filename = '/cluster/work/math/ebeck/data/pacs/art_painting_test.hdf5'
  f = h5py.File(filename, 'r')

  a_group_key = list(f.keys())[0]
  X_train1 = list(f[a_group_key])
  a_group_key = list(f.keys())[1]
  y_train1 = list(f[a_group_key])
  
  # append cartoon training data
  filename = '/cluster/work/math/ebeck/data/pacs/cartoon_train.hdf5'
  f = h5py.File(filename, 'r')

  a_group_key = list(f.keys())[0]
  X_train2 = (list(f[a_group_key]))
  a_group_key = list(f.keys())[1]
  y_train2 = (list(f[a_group_key]))

  # append sketch training data
  filename = '/cluster/work/math/ebeck/data/pacs/sketch_train.hdf5'
  f = h5py.File(filename, 'r')

  a_group_key = list(f.keys())[0]
  X_train3 = (list(f[a_group_key]))
  a_group_key = list(f.keys())[1]
  y_train3 = (list(f[a_group_key]))


  # read in test data

  filename = '/cluster/work/math/ebeck/data/pacs/photo_val.hdf5'
  f = h5py.File(filename, 'r')

  a_group_key = list(f.keys())[0]
  X_test = list(f[a_group_key])
  a_group_key = list(f.keys())[1]
  y_test = list(f[a_group_key])


  # convert data to tensors

  X_train1, X_train2, X_train3 = np.array(X_train1, dtype=np.float32), np.array(X_train2, dtype=np.float32), np.array(X_train3, dtype=np.float32)
  y_train1, y_train2, y_train3  = np.array(y_train1, dtype=np.float32), np.array(y_train2, dtype=np.float32), np.array(y_train3, dtype=np.float32)
  X_test = np.array(X_test, dtype=np.float32)
  y_test = np.array(y_test, dtype=np.float32)
  y_train1, y_train2, y_train3, y_test = y_train1-1, y_train2-1, y_train3-1, y_test-1

  X_train1, X_train2, X_train3 = torch.from_numpy(X_train1).float(), torch.from_numpy(X_train2).float(), torch.from_numpy(X_train3).float()
  y_train1, y_train2, y_train3 = torch.from_numpy(y_train1).float(), torch.from_numpy(y_train2).float(), torch.from_numpy(y_train3).float()
  y_train1, y_train2, y_train3 = F.one_hot(y_train1.to(torch.int64)), F.one_hot(y_train2.to(torch.int64)), F.one_hot(y_train3.to(torch.int64))
  X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
  y_test = F.one_hot(y_test.to(torch.int64))

  # Build environments

  def make_environment(images, labels):
      return{
        'images': (images.float() / 255.).cuda(),
        'labels': labels[:, None].cuda()
      }

  envs = [
    make_environment(X_train1, y_train1),
    make_environment(X_train2, y_train2),
    make_environment(X_train3, y_train3),
    make_environment(X_test, y_test)
  ]


  # Define and instantiate the model

  class MLP(nn.Module):
    def __init__(self):
      super(MLP, self).__init__()
      if flags.grayscale_model:
        lin1 = nn.Linear(14 * 14, flags.hidden_dim)
      else:
        lin0 = models.resnet18(pretrained=True)
        num_ftrs = lin0.fc.in_features
      lin1 = nn.Linear(num_ftrs, flags.hidden_dim)
      lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
      lin3 = nn.Linear(flags.hidden_dim, 7)
      for lin in [lin1, lin2, lin3]:
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
      self._main = nn.Sequential(lin0, nn.ReLU(True), lin1, nn.ReLU(True), 
                                    lin2, nn.ReLU(True), lin3)
    def forward(self, input):
      if flags.grayscale_model:
        out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
      else:
        out = input.permute(0,3,1,2)
      out = self._main(out)
      return out

  mlp = MLP().cuda()

  # Define loss function helpers

  def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y.float())

  def mean_accuracy(logits, y):
    equals = torch.sum(torch.eq(logits.max(1)[1].cuda(),y.max(1)[1].cuda()))
    return equals.float()/logits.shape[0]

  def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

  # Train loop

  def pretty_print(*values):
    col_width = 13
    def format_val(v):
      if not isinstance(v, str):
        v = np.array2string(v, precision=5, floatmode='fixed')
      return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

  optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

  pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

  for step in range(flags.steps):
    for env in envs:
      logits = mlp(env['images'])
      logits = logits.squeeze().float()

      env['labels'] = env['labels'].squeeze()
      env['nll'] = mean_nll(logits, env['labels'])
      env['acc'] = mean_accuracy(logits, env['labels'])
      env['penalty'] = penalty(logits, env['labels'])

    train_nll = torch.stack([envs[0]['nll'], envs[1]['nll'], envs[2]['nll']]).mean()
    train_acc = torch.stack([envs[0]['acc'], envs[1]['acc'], envs[2]['acc']]).mean()
    train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty'], envs[2]['penalty']]).mean()

    weight_norm = torch.tensor(0.).cuda()
    for w in mlp.parameters():
      weight_norm += w.norm().pow(2)

    loss = train_nll.clone()
    loss += flags.l2_regularizer_weight * weight_norm
    penalty_weight = (flags.penalty_weight 
        if step >= flags.penalty_anneal_iters else 1.0)
    loss += penalty_weight * train_penalty
    if penalty_weight > 1.0:
      # Rescale the entire loss to keep gradients in a reasonable range
      loss /= penalty_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    test_acc = envs[3]['acc']
    if step % 100 == 0:
      pretty_print(
        np.int32(step),
        train_nll.detach().cpu().numpy(),
        train_acc.detach().cpu().numpy(),
        train_penalty.detach().cpu().numpy(),
        test_acc.detach().cpu().numpy()
      )

  final_train_accs.append(train_acc.detach().cpu().numpy())
  final_test_accs.append(test_acc.detach().cpu().numpy())
  print('Final train acc (mean/std across restarts so far):')
  print(np.mean(final_train_accs), np.std(final_train_accs))
  print('Final test acc (mean/std across restarts so far):')
  print(np.mean(final_test_accs), np.std(final_test_accs))