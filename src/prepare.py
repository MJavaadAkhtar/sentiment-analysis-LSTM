'''
This file prepares data to be parsed by our model. The data is being fetched
from a csv file, and converted to 3 dataset. 
'''

import csv
import torch
import torch.nn.functional as F
import torchtext
import random
from global_var import *

glove = torchtext.vocab.GloVe(name="6B", dim=50) # Getting all the glove embedding

def get_data():
    '''
    This function loads data from a file. You can change this function to 
    meet your own needs.

    @return: The data from the csv file. 
    '''

    return csv.reader(open("../data/training.1600000.processed.noemoticon.csv", "rt", encoding="latin-1"))



def text_splits(text):
    '''
    This function splits text based on basic punctuation. Four basic 
    punctuations are full stop, comma, semi-colon and question marks
    
    @params text: The input text string 
    @returns: a list of lower case words
    '''
    # separate punctuations
    text = text.replace(".", " . ").replace(",", " , ") \
                 .replace(";", " ; ").replace("?", " ? ")
    return text.lower().split()


def get_words(glove_vector):
    '''
    The function gets all the text sentences and their labels and converts into a 
    torch function so it can processed by the model.

    @params glove_vector: glove embedding vector to lookup word index of the words.
    @returns: list of torch tensor, label tuple for test, train and validation set
    '''

    train, valid, test = [], [], []
    for i, line in enumerate(get_data()):
        if i % 1 == 0:
            tweet = line[-1]
            idxs = [glove_vector.stoi[w]       
                    for w in text_splits(tweet)
                    if w in glove_vector.stoi] 
            if not idxs: # if no word embedding if found for the tweet, ignore
                continue
            idxs = torch.tensor(idxs) # converting list to torch tensor

            if line[0] == "4":
                lb=1
            elif line[0] == "2":
                lb=2
            else:
                lb=0

            label = torch.tensor(int(lb)).long() # changing label to tensor

            if i % 5 < 3:
                train.append((idxs, label)) # 60% is training dataset
            elif i % 5 == 4:
                valid.append((idxs, label)) # 20% is validation dataset
            else:
                test.append((idxs, label)) # 20% is testing dataset
    return train, valid, test


train, valid, test = get_words(glove) # 954177 317964 318072
print(len(train), len(valid), len(test))
