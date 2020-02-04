import torch
import numpy as np
import random
from torch import nn, optim
import copy
import itertools
import util

def mean_accuracy(logits, y):
    preds = ((logits > 0.).float()).cuda()
    accuracy = ((preds.cuda() - y.cuda()).abs() < 1e-2).float().mean()
    return accuracy

def validate_epoch(data_train, data_test, model, loss_function):
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
            model = model.eval()
            outputs = model(inputs)
            # get loss per domain
            loss = loss_function(torch.squeeze(outputs), labels.cuda())
            # get mean loss
            loss_train += loss
            # get accuracy per domain
            accuracy = mean_accuracy(torch.squeeze(outputs), labels)
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
            model = model.eval()
            outputs = model(inputs)
            # get loss per domain
            loss = loss_function(torch.squeeze(outputs), labels.cuda())
            # get mean loss
            loss_test += loss
            # get accuracy per domain
            accuracy = mean_accuracy(torch.squeeze(outputs), labels)
            # append mean accuracy
            accuracy_test += accuracy
    accuracy_test = (accuracy_test/i).detach().cpu().numpy()
    loss_test = (loss_test/i).detach().cpu().numpy()

    return loss_train, accuracy_train, loss_test, accuracy_test


def train_step1(model_task1, model_task2,  input1, input2, optimizer_task1, 
                optimizer_task2,  loss_function):
    inputs1, labels1 = input1
    inputs2, labels2 = input2
    # get loss for model_task1 and apply gradient descent
    model_task1_loss = loss_function(torch.squeeze(model_task1(inputs1)), labels1.cuda())
    # zero the parameter gradients
    optimizer_task1.zero_grad()
    # perform gradient descent
    model_task1_loss.backward()
    optimizer_task1.step()
    # get loss for model_task2 and apply gradient descent                                
    model_task2_loss = loss_function(torch.squeeze(model_task2(inputs2)), labels2.cuda())
    # zero the parameter gradients
    optimizer_task2.zero_grad()
    # perform gradient descent
    model_task2_loss.backward()
    optimizer_task2.step()

def train_step2(model1, model2, model_regularizer, input1, input2,
                loss_function, learning_rate, models, random_domains):
    inputs1, labels1 = input1
    inputs2, labels2 = input2

    optimizer1 = optim.SGD(model1.linear1.parameters(), lr=learning_rate, momentum=0.9)
    optimizer2 = optim.SGD(model2.linear1.parameters(), lr=learning_rate, momentum=0.9)

    # get loss for model_task1 
    model1_loss = loss_function(torch.squeeze(model1(inputs1)), labels1.cuda()) 
    # get loss for model_task2
    model2_loss = loss_function(torch.squeeze(model2(inputs2)), labels2.cuda())  
    # save losses in list                                
    loss = [model1_loss, model2_loss]
    # random meta train loss
    meta_train_loss = loss[random_domains[0]]
    # random meta train model
    meta_train_model = models[random_domains[0]]
    # choose the right optimizer
    optimizer = [optimizer1, optimizer2][random_domains[0]]
    # zero the parameter gradients
    optimizer.zero_grad()
    # perform gradient descent
    meta_train_loss.backward(retain_graph=True)
    optimizer.step()
    # get gradients of regularizer (probably won't work instantly)
    output = model_regularizer(torch.abs(torch.flatten(meta_train_model.linear1.weight)))
    # zero the parameter gradients
    optimizer.zero_grad()
    # perform gradient descent
    meta_train_loss.backward()
    optimizer.step()

def train_step3(model_regularizer, input1, input2, optimizer_reg, 
                loss_function, models ,random_domains):
    # get gradients and apply SGD
    meta_test_model = models[random_domains[1]]
    inputs = [input1, input2][random_domains[1]]
    meta_test_loss = loss_function(torch.squeeze(meta_test_model(inputs[0])), inputs[1].cuda())
    # zero the parameter gradients
    optimizer_reg.zero_grad()
    # perform gradient descent
    meta_test_loss.backward()
    optimizer_reg.step()
  


# train one epoch of the metalearning step (not train the full model)
def train_one_epoch_metatrain(model_task1, model_task2, model_regularizer ,train_input1, 
                    train_input2, optimizer_task1, optimizer_task2, 
                    loss_function, learning_rate, meta_train_steps):

    # TRAIN STEP 1, regular training (line 2-7 in MetaReg algo)
    for i, (input1, input2) in enumerate(zip(train_input1, train_input2)):
        train_step1(model_task1, model_task2, input1, input2, optimizer_task1, 
                    optimizer_task2, loss_function)
                    
    # sample two random domains
    random_domains = random.sample([0, 1], 2)
    # create deepcopies of models for meta learning
    model1 = copy.deepcopy(model_task1)
    model2 = copy.deepcopy(model_task2)
    # make list with models
    models = [model1, model2]
    # choose randomly n metatrain steps
    meta_train_sample = util.sample(zip(train_input1, train_input2), meta_train_steps)
    # TRAIN STEP 2, meta learning of regularizer (line 10-13 in MetaReg algo)
    for input1, input2 in meta_train_sample:
        train_step2(model1, model2, model_regularizer, input1, input2, loss_function, 
                    learning_rate, models=models, random_domains=random_domains)

    optimizer_reg = optim.SGD(model_regularizer.parameters(), lr=learning_rate, momentum=0.9)
    # TRAIN STEP 3, update regularizer NN (line 16 in MetaReg algo)
    for input1, input2 in meta_train_sample:
        train_step3(model_regularizer, input1, input2, 
                optimizer_reg, loss_function, models=models, 
                random_domains=random_domains)

def train_step_full(input, model_final, model_regularizer, loss_function ,optimizer_final):
    inputs, labels = input
    model_final = model_final.train()
    # get classification loss for model_final 
    loss_final_classification = loss_function(torch.squeeze(model_final(inputs)), labels.cuda())
    # get regularization penalty loss                              
    loss_final_regularizer = model_regularizer(torch.abs(torch.flatten(model_final.linear1.weight)))
    # add both losses
    loss_final = loss_final_classification + loss_final_regularizer
    # zero the parameter gradients
    optimizer_final.zero_grad()
    # perform gradient descent
    loss_final.backward()
    optimizer_final.step()

def train_one_epoch_full(input, model_final, model_regularizer, loss_function ,optimizer_final):
    for i, inputs in enumerate(input):
        train_step_full(inputs, model_final, model_regularizer, loss_function ,optimizer_final)
