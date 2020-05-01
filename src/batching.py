'''
This class creates mini-batches from within the dataset.
'''

import torch
import random


class BatchGenerator:
    def __init__(self, text, batch_size=32, drop_last=False):
        self.text_by_length = {}
        for words, label in text:
            wlen = words.shape[0]
            if wlen not in self.text_by_length:
                self.text_by_length[wlen] = []
            self.text_by_length[wlen].append((words, label),)
        self.loaders = {wlen : torch.utils.data.DataLoader(
                                    text,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    drop_last=drop_last) 
            for wlen, text in self.text_by_length.items()}
        
    def __iter__(self): # Create an iterator for the class to be used
        iters = [iter(loader) for loader in self.loaders.values()]
        while iters:
            im = random.choice(iters)
            try:
                yield next(im)
            except StopIteration:
                iters.remove(im)