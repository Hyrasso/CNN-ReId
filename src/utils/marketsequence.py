import os
from h5py import File
from keras.utils import Sequence
import numpy as np
import json

class MarketSequence():
    def __init__(self, market_path, batch_size, preprocess=None, train=True):
        self.batch_size = batch_size
        self.f = File(market_path, "r")
        if train:
            self.x = self.f["train_images"]
            self.y = self.f["train_labels"]
        else:
            self.x = self.f["test_images"]
            self.y = self.f["test_labels"]

        if isinstance(preprocess, tuple):
            assert len(preprocess) == 2
            self.preprocess_input, self.preprocess_output = preprocess

        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, index):
        x = np.array(self.x[self.batch_size * index:self.batch_size * (index + 1)])
        y = np.array(self.y[self.batch_size * index:self.batch_size * (index + 1)])
        # print(x)
        if hasattr(self, "preprocess_input"):
            x = self.preprocess_input(x)
        if hasattr(self, "preprocess_output"):
            y = self.preprocess_output(y)
        return x, y


if __name__ == "__main__":
    import time

    ms = MarketSequence("market.h5", 32, train=False)
    batch = ms[0]
    x, y = batch
    assert x.shape[0] == 32, "Batch size"
    assert y.shape[0] == 32, "Batch size"
    assert len(x.shape[1:]) == 3, "RGB images"
    assert y.shape[1] == 27, "Number of attributes"

    def identity(a):
        return a
    
    ms = MarketSequence("market.h5", 16, train=True, preprocess=(identity, identity))
    batch = ms[2]
    x, y = batch
    assert x.shape[0] == 16, "Batch size"
    assert y.shape[0] == 16, "Batch size"
    assert len(x.shape[1:]) == 3, "RGB images"
    assert y.shape[1] == 27, "Number of attributes"
    l = []
    for i in range(len(ms)):
        l.append(ms[i])
    print("Loaded all train data in memory")
    

