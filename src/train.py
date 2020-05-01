'''
This file contains function that conducts training and checks accuracy  of the model.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import numpy as np
import matplotlib.pyplot as plt
from model import SentimentAnalysis
from prepare import *
from batching import BatchGenerator


def train_rnn_network(model, train, valid, num_epochs=5, learning_rate=1e-5, checkpoint_path=None):
    '''
    The function trains the model based on the parameters given above and plot
    the end results.

    @params model : The model that needs to be trained
    @params train : The training dataset
    @params valid : The validation dataset
    @params num_epoch : Number of epochs
    @params learning_rate : Learning rate that is being used
    @params checkpoint_path : file path name where checkpoint is going to be saved
    @return: None
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    for epoch in range(num_epochs):
        for texts, labels in train:
            optimizer.zero_grad()
            pred = model(texts)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        losses.append(float(loss))

        epochs.append(epoch)
        train_acc.append(get_accuracy(model, train_loader))
        valid_acc.append(get_accuracy(model, valid_loader))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
              epoch+1, loss, train_acc[-1], valid_acc[-1]))
        
        if (checkpoint_path is not None) and epoch % 20 == 0:
            torch.save(model.state_dict(), checkpoint_path.format(epoch))
    # plotting
    plt.title("Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, valid_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

def get_accuracy(model, data_loader):
    '''
    The function calculates the accuracy of the model against by the dataset batch
    provided.

    @params model: The model that is being trained
    @params data_loader: The batch we are testing the accuracy against
    @return: percentage accuracy
    '''
    model.eval()
    correct, total = 0, 0
    for texts, labels in data_loader:
        output = model(texts)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total


model = SentimentAnalysis(50, 50, 3)
train_loader = BatchGenerator(train, batch_size=64, drop_last=True) # 64 before
valid_loader = BatchGenerator(valid, batch_size=64, drop_last=False)
test_loader = BatchGenerator(test, batch_size=64, drop_last=False)
train_rnn_network(model, train_loader, valid_loader, num_epochs=3, learning_rate=2e-5, checkpoint_path='./RNN_50_50_64_{}')
