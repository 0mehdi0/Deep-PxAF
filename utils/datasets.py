import os
import csv
import cv2
import math
import copy
import torch
import timeit
import pickle
import random
import platform
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from subprocess import check_output
from torchvision.datasets import CIFAR10
from biosppy.signals import tools as tools
from signalprocess_1ch import signal_process

xlen2=0
xlen=0
###########################    def Data_set  ###############################
def Data_set():
  labels = []
  if os.path.exists('datasets/Selected_PAF.csv'):
    with open('datasets/Selected_PAF.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(row)     # set labels
  else :
    labels = [["num", "label"]]
    for index in range(1875):
        labels.append([index,0])   # set labels
      
  labeled_data = pd.read_pickle('datasets/dataset_PAF.pkl')
  mylist=labeled_data.values.tolist()

  ch=0
  PAF_sig=[]
  for index in range(len(mylist)):
      if mylist[index][ch][0] == 'p' and int(mylist[index][ch][1:3])%2==0 :
          if mylist[index][ch][-1] == 'c': #PAF 5min
            #print(mylist[index][ch])
            PAF_sig.append(mylist[index])
  #fix some missing value in data set
  for index in [1,20,21,22]:
    len_=len(PAF_sig[index][1])-1
    PAF_sig[index][1][len_] = PAF_sig[index][1][len_-1]
    PAF_sig[index][2][len_] = PAF_sig[index][2][len_-1]
  filter_ch1 = []
  filter_ch2 = []
  #filtering data
  sampling_rate = 128
  for i in range(len(PAF_sig)): 
    for ch in range(1,3):
      signal = np.array(PAF_sig[i][ch])
      order = int(0.3 * sampling_rate)
      filtered, _, _ = tools.filter_signal(signal=signal,
                                    ftype='FIR',
                                    band='bandpass',
                                    order=order,
                                    frequency=[3, 45],
                                    sampling_rate=sampling_rate)
      if ch==1:                              
        filter_ch1.append(filtered)
      else :
        filter_ch2.append(filtered)

  data_ch1 = []
  data_ch2 = []
  data_ = []
  len_signal = 512
  l_=[]

  for item in labels:
    if item[0]!='':
        l_.append(item)

  labels = l_   

  for i in range(len(PAF_sig)):
    for j in range(int(len(filter_ch1[i])/len_signal)):
      if labels[i*75 + j + 1][1] == '1':
        arr1 = np.array(filter_ch1[i][j*len_signal:(j+1)*len_signal])
        sig1 =   (2*(arr1 - np.min(arr1))/np.ptp(arr1))-1    
        arr2 = np.array(filter_ch2[i][j*len_signal:(j+1)*len_signal])
        sig2 =   (2*(arr2 - np.min(arr2))/np.ptp(arr2))-1   
        data_ch1.append(sig1)
        data_ch2.append(sig2)
        data_.append(sig1)
        data_.append(sig2)

 
  data = torch.load('datasets/GAN_Data.pt')
  indexes = torch.load("datasets/Selected_GAN.pt")
  paf_gan=[]
  for i in indexes:
    paf_gan.append(np.array(data[i],dtype='float'))
  paf_gan = np.array(paf_gan,dtype='float')
  paf_gan = (torch.from_numpy(paf_gan))
  data_ch1=np.array(data_ch1)
  data_ch2=np.array(data_ch2)
  data_ = np.array(data_,dtype='float')
  data_ = (torch.from_numpy(data_))
  return data_, paf_gan
########################    end of def Data_set  ###########################

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
    X_ = np.append(Xtr,Xte,axis=0)
    Y_ = np.append(Ytr,Yte,axis=0)
    
    data_normal=[]
    for item in range(len(X_)):
        if Y_[item] == 0:
            data_normal.append(X_[item])

    data_paf, paf_gan = Data_set()
    data_paf_ = []
    wavelet_type="db3"
    rm_coefficients=[1,6,7,8,9,10]
    window_len=10
    recurrence_eps=0.001
    sam = 512

    for item in data_paf:
      x,x,x,x,x,x,x,rec1=signal_process(item,sam,wavelet_type,\
                                  rm_coefficients,window_len,recurrence_eps)
      # resize to 100*100
      rec1 = cv2.resize(rec1, (100,100))
      data_paf_.append(([np.array(rec1)])) 

    data_paf_gan_=[]

    for item in paf_gan:
      x,x,x,x,x,x,x,rec1=signal_process(item,sam,wavelet_type,\
                                  rm_coefficients,window_len,recurrence_eps)

      rec1 = cv2.resize(rec1, (100,100))
      data_paf_gan_.append(([np.array(rec1)])) 

    data_paf_gan = np.array(data_paf_gan_)
    data_paf_gan = torch.from_numpy(data_paf_gan).float()
    data_paf = data_paf_
    data_paf = np.array(data_paf)
    data_paf = torch.from_numpy(data_paf).float()
    data_normal_ = []

    print('len data_normal:',len(data_normal))
    print('len data_paf:',len(data_paf))
    print('len data_paf:',len(data_paf_gan))

    data_normal = np.array(data_normal)
    data_normal = torch.from_numpy(data_normal).float()
    return data_normal, data_paf, data_paf_gan

def get_ECG_data(seed, GAN_flag):
    random.seed(seed)
    ECG_dir = 'datasets/ECG3class_1ch'
    X_normal, X_paf , data_paf_gan = load_ECG(ECG_dir) 
    Y_normal = []
    Y_paf =[]
    Y_paf_gan = []
    for i in range(len(X_normal)):
        Y_normal.append(0)
    for i in range(len(X_paf)):
        Y_paf.append(1)
    for i in range(len(data_paf_gan)):
        Y_paf_gan.append(1)
    data_normal=(list(zip(X_normal,Y_normal))) 
    data_paf=(list(zip(X_paf,Y_paf))) 
    data_paf_gan = (list(zip(data_paf_gan,Y_paf_gan))) 
    data_normal.extend(data_paf)
    all_data=data_normal
    all_data_gan = copy.deepcopy(all_data)
    all_data_gan.extend(data_paf_gan)
    num_train = int(len(all_data)*0.7)
    num_val_test = int(len(all_data)*0.15) 
    random.shuffle(all_data)
    test_data = all_data[:num_val_test]
    all_data = all_data[num_val_test:]
    if GAN_flag :
      all_data.extend(data_paf_gan)
    random.shuffle(all_data)
    trian_data = all_data[len(test_data):]
    val_data   = all_data[:len(test_data)]
    print("num train val test: ", len(trian_data),len(val_data),len(test_data))

    return trian_data,val_data,test_data

if __name__ == "__main__":
    ECG_dir = 'datasets/ECG3class_1ch'
    X_normal, X_paf,_ = get_ECG_data()
    print(len(X_normal),len(X_paf))

