import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as signal_
from biosppy.signals import tools as tools

def Data_set():
  labeled_data = pd.read_pickle('dataset/dataset_PAF.pkl')
  mylist=labeled_data.values.tolist()
  ch=0
  PAF_sig=[]
  
  for index in range(len(mylist)):
      if mylist[index][ch][0] == 'p' and int(mylist[index][ch][1:3])%2==0 :
          if mylist[index][ch][-1] == 'c': #PAF 5min
            #print(mylist[index][ch])
            PAF_sig.append(mylist[index])
            
  # fix some missing value
  for index in [1,20,21,22]:
    len_=len(PAF_sig[index][1])-1
    PAF_sig[index][1][len_] = PAF_sig[index][1][len_-1]
    PAF_sig[index][2][len_] = PAF_sig[index][2][len_-1]

  filter_ch1 = []
  filter_ch2 = []
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
  len_signal = 5000
  
  for i in range(len(PAF_sig)):
    for j in range(int(len(filter_ch1[i])/len_signal)):
      #print(i)
      arr1 = np.array(filter_ch1[i][j*len_signal:(j+1)*len_signal])
      sig1 =   (2*(arr1 - np.min(arr1))/np.ptp(arr1))-1    #PAF_sig[i][2]
      arr2 = np.array(filter_ch2[i][j*len_signal:(j+1)*len_signal])
      sig2 =   (2*(arr2 - np.min(arr2))/np.ptp(arr2))-1    #PAF_sig[i][2]
      sig1=signal_.resample(sig1, 5000)
      sig2=signal_.resample(sig2, 5000)
      if (np.where(np.isnan(sig1) == True)[0])!=[]:
        print("nan",i,j)
      else:
        data_ch1.append(sig1)
        data_ch2.append(sig2)
        data_.append([sig1,sig2])
  data_ch1=np.array(data_ch1)
  data_ch2=np.array(data_ch2)
  data_ = np.array(data_,dtype='float')
  np.random.shuffle(data_)
  data_ = (torch.from_numpy(data_))
  return data_
