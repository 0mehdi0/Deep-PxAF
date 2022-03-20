# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10





import tensorflow as tf
import pandas as pd
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import platform
from subprocess import check_output
xlen2=0
xlen=0




def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))



def load_ECG_batch(filename):
    """ load single batch of ECG """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)       
        X = datadict['data']
        Y = datadict['labels']
        #Y=list(np.array(Y)-1)
        xlen = datadict['len']
        print((X.shape),len(X),type(X))
        X=X.reshape(xlen,1,100,100)

        X = np.array(X)
        Y = np.array(Y)
        return X, Y,xlen
def load_ECG(ROOT):
    """ load all of ECG """
    xs = []
    ys = []
    xlen2=0
    xlen=0
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y,xlen = load_ECG_batch(f)
        xlen2=xlen2+xlen
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte,xlen = load_ECG_batch(os.path.join(ROOT, 'test_batch'))
    xlen2=xlen2+xlen   
    return Xtr, Ytr, Xte, Yte ,xlen2
def get_ECG_data(num_training=5000, num_validation=0, num_test=1000):
    # Load the raw ECG-10 data
    ECG_dir = './datasets/ECG3class_1ch'
    X_train, y_train, X_test, y_test,xlen = load_ECG(ECG_dir)
    #num_training=int((xlen/6)*5)
    #num_test=int(xlen/6)
    print(len(X_test))
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    #X_test=X_test[0]
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 5
    x_train=x_train-1
    x_test /= 5
    x_test=x_test-1
    x_train=torch.from_numpy(x_train)
    xnew=(list(zip(x_train,y_train)))
    x_test=torch.from_numpy(x_test)
    xtestnew=(list(zip(x_test,y_test)))
    return xnew,xtestnew






class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def get_dataset(cls, cutout_length=0):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    cutout = []
    if cutout_length > 0:
        cutout.append(Cutout(cutout_length))

    train_transform = transforms.Compose(transf + normalize + cutout)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid
