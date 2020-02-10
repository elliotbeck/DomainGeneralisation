import torch
from torch import nn, optim
import numpy as np
import util
import itertools
import copy


def loss_fn_task(input1, input2, feature_network, task_network, loss_function):
    inputs1, labels1 = input1
    inputs2, labels2 = input2

    # concat the features and labels from training domains, take the first two.
    inputs = torch.cat([inputs1, inputs2], 0)
    labels = torch.cat([labels1, labels2], 0)

    # get outputs of feature_network
    feature_network_output = feature_network(inputs)
    # get outputs of task_network
    task_network_output = task_network(feature_network_output)
    # calculate mean loss over the training domains
    task_network_loss = loss_function(task_network(feature_network(inputs)), 
                    torch.tensor(torch.squeeze(labels), dtype=torch.long).cuda())
    return task_network_loss


def loss_fn_global(input1, input2, input3, feature_network, task_network):
    
    inputs1, labels1 = input1
    inputs2, labels2 = input2
    inputs3, labels3 = input3

    # seperate the features contained in domain 1 into different classes
    class1_domain1 = []
    class2_domain1 = []
    class3_domain1 = []
    class4_domain1 = []
    class5_domain1 = []
    class6_domain1 = []
    class7_domain1 = []

    features_classes_domain1 = [class1_domain1, class2_domain1, class3_domain1, class4_domain1,
                        class5_domain1, class6_domain1, class7_domain1]
    for c in range(0,7):
        for index, input in enumerate(inputs1):
            if labels1[index] == c:
                features_classes_domain1[c].append(input)
    # seperate the features contained in domain 2 into different classes
    class1_domain2 = []
    class2_domain2 = []
    class3_domain2 = []
    class4_domain2 = []
    class5_domain2 = []
    class6_domain2 = []
    class7_domain2 = []

    features_classes_domain2 = [class1_domain2, class2_domain2, class3_domain2, class4_domain2,
                        class5_domain2, class6_domain2, class7_domain2]

    for c in range(0,7):
        for index, input in enumerate(inputs2):
            if labels2[index] == c:
                features_classes_domain2[c].append(input)

    # seperate the features contained in domain 3 into different classes
    class1_domain3 = []
    class2_domain3 = []
    class3_domain3 = []
    class4_domain3 = []
    class5_domain3 = []
    class6_domain3 = []
    class7_domain3 = []

    features_classes_domain3 = [class1_domain3, class2_domain3, class3_domain3, class4_domain3,
                        class5_domain3, class6_domain3, class7_domain3]

    for c in range(0,7):
        for index, input in enumerate(inputs3):
            if labels3[index] == c:
                features_classes_domain3[c].append(input)

    # get the mean of outputs per class per domain of the model (eq. 2)
    # domain 1
    domain1_output = []
    for c in range(0,7):
        inputs = features_classes_domain1[c]
        if len(inputs) == 0:
            domain1_output.append(torch.zeros(1024).cuda())
        else:
            inputs = torch.stack(inputs)
            domain1_output.append(feature_network(inputs))
            domain1_output[c] = torch.mean(domain1_output[c], dim=0)

    # domain 2
    domain2_output = []
    for c in range(0,7):
        inputs = features_classes_domain2[c]
        if len(inputs) == 0:
            domain2_output.append(torch.zeros(1024).cuda())
        else:
            inputs = torch.stack(inputs)
            domain2_output.append(feature_network(inputs))
            domain2_output[c] = torch.mean(domain2_output[c], dim=0)
    
    # domain 3
    domain3_output = []
    for c in range(0,7):
        inputs = features_classes_domain3[c]
        if len(inputs) == 0:
            domain3_output.append(torch.zeros(1024).cuda())
        else:
            inputs = torch.stack(inputs)
            domain3_output.append(feature_network(inputs))
            domain3_output[c] = torch.mean(domain3_output[c], dim=0)

    # get softmax outputs (eq. 3)
    domain1_output_softmax = []
    domain2_output_softmax = []
    domain3_output_softmax = []  

    for c in range(0,7):
        domain1_output_softmax.append(torch.nn.functional.softmax(task_network(domain1_output[c]) / 2.).data) # TODO tau as variable
        domain2_output_softmax.append(torch.nn.functional.softmax(task_network(domain2_output[c]) / 2.).data) # TODO tau as variable
        domain3_output_softmax.append(torch.nn.functional.softmax(task_network(domain3_output[c]) / 2.).data) # TODO tau as variable
    # calculate the loss function 
    loss_global1 = []
    loss_global2 = [] 
    for c in range(0,7):
        loss_global1.append(util.kd(domain1_output_softmax[c], domain3_output_softmax[c]))
        loss_global2.append(util.kd(domain2_output_softmax[c], domain3_output_softmax[c]))
    loss_global1_mean = torch.mean(torch.stack(loss_global1))
    loss_global2_mean = torch.mean(torch.stack(loss_global2))
    loss_global = torch.mean(torch.stack([loss_global1_mean, loss_global2_mean]))
    return loss_global

def loss_fn_local(input1, input2, input3, embedding_network, eps):
    # get inputs and labels
    inputs1, labels1 = input1
    inputs2, labels2 = input2
    inputs3, labels3 = input3
    inputs = torch.cat((torch.cat((inputs1, inputs2)), inputs3))
    labels = torch.cat((torch.cat((labels1, labels2)), labels3))
    # get the embedding vectors 
    embeddings = torch.squeeze(embedding_network(inputs))

    # initialize loss instance
    loss_triplet = util.TripletLoss(margin=eps)

    # initialize triplet selector
    selector = util.BatchHardTripletSelector()

    # get the triplets 
    anchor, pos, neg = selector(embeddings, labels)

    # calculate the loss 
    loss_local = loss_triplet(anchor, pos, neg)
    
    return loss_local

def _train_step1(feature_network_copy, task_network_copy, input1, input2,
                 optimizer_feature_copy, optimizer_task_copy, loss_function):

    # get loss of classifier
    loss = loss_fn_task(input1, input2, feature_network_copy, task_network_copy, loss_function)

    # zero the parameter gradients, update task network
    optimizer_task_copy.zero_grad()
    # perform gradient descent
    loss.backward()
    optimizer_task_copy.step()

    # get loss of classifier
    loss = loss_fn_task(input1, input2, feature_network_copy, task_network_copy, loss_function)

    # zero the parameter gradients, update feature network
    optimizer_feature_copy.zero_grad()
    # perform gradient descent
    loss.backward()
    optimizer_feature_copy.step()
    


def _train_step2(feature_network, feature_network_copy, task_network, task_network_copy, 
                embedding_network, input1, input2, input3, optimizer_feature, 
                optimizer_task, optimizer_embedding, eps, loss_function):


    # get loss of critic
    loss_global = loss_fn_global(input1, input2, input3, feature_network_copy, 
                                task_network_copy)
    loss_local = loss_fn_local(input1, input2, input3, embedding_network, eps)
    loss_meta = loss_global + 0.005 * loss_local
    loss_task = loss_fn_task(input1, input2, feature_network_copy, 
                            task_network_copy, loss_function)
    # get loss to update parameters
    loss_critic = loss_meta + loss_task

    # other approach feature network updates
    feature_network_copy.zero_grad()
    task_network_copy.zero_grad()
    loss_critic.backward(retain_graph=True)
    with torch.no_grad():
        for p, q in zip(feature_network.parameters(), feature_network_copy.parameters()):
            new_val = p - 0.001*q.grad
            p = new_val

        for p, g in zip(task_network.parameters(), task_network_copy.parameters()):
            new_val = p - 0.001*g.grad
            p.copy_(new_val)




    # # update parametersof feature network
    # # zero the parameter gradients, update feature network
    # optimizer_feature.zero_grad()
    # # perform gradient descent
    # loss_critic.backward(retain_graph=True)
    # optimizer_feature.step()

    # # update parameters of task network
    # # zero the parameter gradients, update feature network
    # optimizer_task.zero_grad()
    # # perform gradient descent
    # loss_critic.backward()
    # optimizer_task.step()

    # update parameters of embedding network
    loss_local = loss_fn_local(input1, input2, input3, embedding_network, eps)
    # zero the parameter gradients, update feature network
    optimizer_embedding.zero_grad()
    # perform gradient descent
    loss_local.backward()
    optimizer_embedding.step()


def train_one_epoch(feature_network, task_network, embedding_network, train_input1, train_input2, 
                    train_input3, train_input_full, optimizer_feature, optimizer_task, optimizer_embedding, eps, 
                    learning_rate, loss_function):
    # set model status to train
    feature_network = feature_network.train()
    task_network = task_network.train()

    # get a copy of original model
    # feature_network_copy = copy.deepcopy(feature_network) 
    # task_network_copy = copy.deepcopy(task_network)

    feature_network_copy = type(feature_network)(1024).cuda() # get a new instance
    feature_network_copy.load_state_dict(feature_network.state_dict()) # copy weights and stuff

    task_network_copy = type(task_network)(feature_network, 1024, 7).cuda() # get a new instance
    task_network_copy.load_state_dict(task_network_copy.state_dict()) # copy weights and stuff

    # set optimizers for copied networks
    optimizer_feature_copy = optim.SGD(feature_network_copy.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_task_copy = optim.SGD(task_network_copy.parameters(), lr=learning_rate, momentum=0.9)

    for input1, input2, _ in zip(train_input1, train_input2, train_input3):
        _train_step1(feature_network_copy, task_network_copy, input1, input2,
                     optimizer_feature_copy, optimizer_task_copy, loss_function)

    for input1, input2, input3 in zip(train_input1, train_input2, train_input3):
        _train_step2(feature_network, feature_network_copy, task_network, task_network_copy, 
                    embedding_network, input1, input2, input3 , optimizer_feature, 
                    optimizer_task, optimizer_embedding, eps, loss_function)

# define accuracy function 
def mean_accuracy(logits, y):
   probs = torch.softmax(logits, dim=1)
   winners = probs.argmax(dim=1)
   corrects = (winners.unsqueeze(-1) == y.cuda())
   accuracy = corrects.sum().float() / float(y.size(0))
   return accuracy

def validate_epoch(data_train, data_test, model_feature, model_task, loss_function):
    loss_test = 0
    accuracy_test = 0
    loss_train = 0
    accuracy_train = 0
    # get accuracy and loss on train data
    with torch.no_grad():
        for i, batch in enumerate(data_train, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            # forward
            model_feature = model_feature.eval()
            model_task = model_task.eval()
            outputs = model_feature(inputs)
            outputs = model_task(outputs)
            # get loss per domain
            loss = loss_function(outputs, torch.tensor(torch.squeeze(labels)
            , dtype=torch.long).cuda())
            # get mean loss
            loss_train += loss
            # get accuracy per domain
            accuracy = mean_accuracy(outputs, labels)
            # append mean accuracy
            accuracy_train += accuracy
    accuracy_train = (accuracy_train/i).detach().cpu().numpy()
    loss_train = (loss_train/i).detach().cpu().numpy()

    # get accuracy and loss on test data
    with torch.no_grad():
        for i, batch in enumerate(data_test, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            # forward
            model_feature = model_feature.eval()
            model_task = model_task.eval()
            outputs = model_feature(inputs)
            outputs = model_task(outputs)
            # get loss per domain
            loss = loss_function(outputs, torch.tensor(torch.squeeze(labels), dtype=torch.long).cuda())
            # get mean loss
            loss_test += loss
            # get accuracy per domain
            accuracy = mean_accuracy(outputs, labels)
            # append mean accuracy
            accuracy_test += accuracy
    accuracy_test = (accuracy_test/i).detach().cpu().numpy()
    loss_test = (loss_test/i).detach().cpu().numpy()

    return loss_train, accuracy_train, loss_test, accuracy_test