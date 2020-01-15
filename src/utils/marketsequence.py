import os
from h5py import File
from keras.utils import Sequence
import numpy as np
import json

class MarketSequence(Sequence):
    def __init__(self, market_path, batch_size, preprocess=None, resize=None, train=True):
        self.batch_size = batch_size
        self.f = File(market_path, "r")
        if train:
            self.x = self.f["train_images"]
            self.y = self.f["train_labels"]
        else:
            self.x = self.f["test_images"]
            self.y = self.f["test_labels"]

        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # TODO: preprocess and resize
        return (
            self.x[self.batch_size * index:self.batch_size * (index + 1)],
            self.y[self.batch_size * index:self.batch_size * (index + 1)]
        )


if __name__ == "__main__":
    import time

    ms = MarketSequence("market.h5", 32, train=False)
    batch = ms[0]
    x, y = batch
    assert x.shape[0] == 32, "Batch size"
    assert y.shape[0] == 32, "Batch size"
    assert len(x.shape[1:]) == 3, "RGB images"
    assert y.shape[1] == 27, "Number of attributes"

    ms = MarketSequence("market.h5", 16, train=True)
    batch = ms[0]
    x, y = batch
    assert x.shape[0] == 16, "Batch size"
    assert y.shape[0] == 16, "Batch size"
    assert len(x.shape[1:]) == 3, "RGB images"
    assert y.shape[1] == 27, "Number of attributes"
    l = []
    for i in range(len(ms)):
        l.append(ms[i])
    print("Loaded all train data in memory")
    

