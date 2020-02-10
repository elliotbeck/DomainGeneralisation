import torch
import random
from torch import nn, optim
import argparse
import numpy as np
from models import model_feature, model_task, model_embedding
from torch.utils import data
from data_loader import HDF5Dataset
from train import validate_epoch, train_one_epoch

#from train import validate_epoch, train_one_epoch_metatrain, train_one_epoch_full

parser = argparse.ArgumentParser(description='PACS')
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--eps', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=1)
flags = parser.parse_args()

#print setup
print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

# set seed
random.seed(flags.seed)

# load data
dataset1 = HDF5Dataset('/cluster/work/math/ebeck/data/pacs/art_painting_train.hdf5')
train_data1 = data.DataLoader(dataset1, num_workers=1, batch_size=flags.batch_size, 
                              shuffle=True, drop_last=True)
dataset2 = HDF5Dataset('/cluster/work/math/ebeck/data/pacs/sketch_train.hdf5')
train_data2 = data.DataLoader(dataset2, num_workers=1, batch_size=flags.batch_size, 
                              shuffle=True, drop_last=True)
dataset3 = HDF5Dataset('/cluster/work/math/ebeck/data/pacs/photo_train.hdf5')
train_data3 = data.DataLoader(dataset3, num_workers=1, batch_size=flags.batch_size, 
                              shuffle=True, drop_last=True)
train_data_full = data.DataLoader(data.ConcatDataset([dataset1, dataset2, dataset3]), 
                                                    num_workers=1, batch_size=flags.batch_size, 
                                                    shuffle=True, drop_last=True)
dataset = HDF5Dataset('/cluster/work/math/ebeck/data/pacs/cartoon_test.hdf5')
test_data = data.DataLoader(dataset, num_workers=1, batch_size=flags.batch_size, 
                              shuffle=True, drop_last=True)

# load models
feature_network = model_feature(flags.hidden_dim).cuda()
task_network = model_task(feature_network, flags.hidden_dim, flags.num_classes).cuda()
embedding_network = model_embedding(feature_network, flags.hidden_dim).cuda()

# set train function 
def trainer(feature_network, task_network, embedding_network, train_data_full, train_data1, train_data2, 
            train_data3, test_data, epochs, learning_rate, eps):
    # set loss function for all NNs
    loss_function = nn.CrossEntropyLoss()

    # set optimizers for metatraining
    optimizer_feature = optim.SGD(feature_network.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_task = optim.SGD(task_network.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_embedding = optim.SGD(embedding_network.parameters(), lr=learning_rate, momentum=0.9)
    

    # metatraining
    for epoch in range(epochs):  

        # shuffle the three source domains
        random = np.array([0, 1, 2])
        np.random.shuffle(random)
        train_domain_data = [train_data1, train_data2, train_data3]
        train_input1, train_input2, train_input3 = train_domain_data[random[0]], train_domain_data[random[1]], train_domain_data[random[2]]
        # train one epoch
        train_one_epoch(feature_network, task_network, embedding_network, train_input1, train_input2, 
                                    train_input3, optimizer_feature, optimizer_task, optimizer_embedding, eps, 
                                    learning_rate, loss_function)

        # validate epoch on validation set
        loss_train, accuracy_train, loss_test, accuracy_test = validate_epoch(train_data_full, test_data, feature_network, 
                                                                    task_network, loss_function)

        # print the metrics
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                                np.array2string(loss_train, precision=2, floatmode='fixed'),
                                np.array2string(accuracy_train*100, precision=2, floatmode='fixed'),
                                np.array2string(loss_test, precision=2, floatmode='fixed'),
                                np.array2string(accuracy_test*100, precision=2, floatmode='fixed')))          
                
    print('Finished Training')

if __name__ == "__main__":
   trainer(feature_network, task_network, embedding_network, train_data_full, train_data1, train_data2, 
            train_data3, test_data, flags.epochs, flags.lr, flags.eps)