from model import SentimentAnalysis
from prepare import *
from batching import BatchGenerator


model_load = SentimentAnalysis(50,50,2)
check = torch.load('./RNN1_300') #Location of the save checkpoint file
model_load.load_state_dict(check)
model_load.eval()
tweet = 'this is cool. i love it.'
glove_vector=glove
idxs = [glove_vector.stoi[w]        # lookup the index of word
        for w in text_splits(tweet)
        if w in glove_vector.stoi] # keep words that has an embedding
idxs = torch.tensor(idxs)
label = torch.tensor(int("4"=="4")).long()
temp = [(idxs,label)]

test = BatchGenerator(temp, batch_size=1, drop_last=False)

def get_accuracy(model, tweet):
    for tweets, labels in  tweet:
        output = model(tweets)
        pred = output.max(1,keepdim=True)[1]
        pred = pred.detach().numpy()
        return (pred == 1).sum()

print(get_accuracy(model_load, test))