from model import SentimentAnalysis
from prepare import *
from batching import TweetBatcher
# from train import get_accuracy
# glove = torchtext.vocab.GloVe(name="6B", dim=50)
model_load = SentimentAnalysis(50,50,2)
check = torch.load('./RNN1_50_50_45')
model_load.load_state_dict(check)
model_load.eval()
tweet = 'this is cool. i love it.'
glove_vector=glove
idxs = [glove_vector.stoi[w]        # lookup the index of word
        for w in split_tweet(tweet)
        if w in glove_vector.stoi] # keep words that has an embedding
idxs = torch.tensor(idxs)
label = torch.tensor(int("4"=="4")).long()
temp = [(idxs,label)]
# test_loader = TweetBatcher(test, batch_size=64, drop_last=False)
# print(train[0])
test = TweetBatcher(temp, batch_size=1, drop_last=False)

def get_accuracy(model, tweet):
    for tweets, labels in  tweet:
        output = model(tweets)
        pred = output.max(1,keepdim=True)[1]
        pred = pred.detach().numpy()
        return (pred == 1).sum()

print(get_accuracy(model_load, test))