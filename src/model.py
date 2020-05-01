'''
This file contains the LSTM Model that is trained for sentiment analysis. There are three major processes:
    1. Embedding creation
    2. LSTM training
    3. Fully connected layer to consolidate data
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

glove = torchtext.vocab.GloVe(name="6B", dim=50) # Getting all the glove embedding

class SentimentAnalysis(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SentimentAnalysis, self).__init__()
        self.hidden_size = hidden_size
        self.embeding = nn.Embedding.from_pretrained(glove.vectors)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self,x):
        x = self.embeding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:,-1,:])
        return out