#signal process

import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
from mne.filter import filter_data, resample
from scipy.signal import detrend, find_peaks,hilbert, cheby1, filtfilt
import numpy as np
from bokeh.plotting import figure, output_file, show
import pandas as pd
import os
import matplotlib.pyplot as plt
import pdb
import pywt
import pickle
import random
import urllib.request
import math
from pyts.datasets import load_gunpoint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField

labeled_data = pd.read_pickle('datasets/dataset_PAF.pkl')
mylist=labeled_data.values.tolist()





def data_preprocess():
  i=0
  ii=1
  for i in range(len(mylist[i])) :
    for ii in range(2):
      diff=mylist[0][1][len(mylist[i][ii])-2]-mylist[0][1][len(mylist[i][ii])-1]
      if diff>2 :
        mylist[0][1][len(mylist[i][ii])-1]=mylist[0][1][len(mylist[i][ii])-2]
  return mylist
#######################################################################################################################
def mtf_plot(s, eps=None, steps=None):
  mtf = MarkovTransitionField(image_size=149)
  ii=0
  ch=s
  X_mtf = mtf.fit_transform(ch[0:1])
  plt.figure(figsize=(5, 5))
  plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
  plt.title('Markov Transition Field', fontsize=18)
  plt.colorbar(fraction=0.0457, pad=0.04)
  plt.tight_layout()
  plt.show()
  return "z"
#######################################################################################################################
def gaf_plot(s, eps=None, steps=None):
  X=s
 # Transform the time series into Gramian Angular Fields
  gasf = GramianAngularField(image_size=128, method='summation')
  X_gasf = gasf.fit_transform(X[0:1])
  gadf = GramianAngularField(image_size=128, method='difference')
  X_gadf = gadf.fit_transform(X[0:1])
  # Show the images for the first time series
  fig = plt.figure(figsize=(8, 4))
  grid = ImageGrid(fig, 111,nrows_ncols=(1, 2),axes_pad=0.15,share_all=True,\
                   cbar_location="right",cbar_mode="single",cbar_size="7%",cbar_pad=0.3,)
  images = [X_gasf[0], X_gadf[0]]
  titles = ['Summation', 'Difference']
  for image, title, ax in zip(images, titles, grid):
    im = ax.imshow(image, cmap='rainbow', origin='lower')
    ax.set_title(title, fontdict={'fontsize': 12})
  ax.cax.colorbar(im)
  ax.cax.toggle_label(True)
  plt.suptitle('Gramian Angular Fields', y=0.98, fontsize=16)
  plt.show()
  return 'Z'
#######################################################################################################################

def find(data,name):
  for k in range(len(data)):
    if data[k][0]==name:
      return k
#######################################################################################################################
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
#######################################################################################################################
def recurrence_fn(s, eps=None, steps=None):
    if eps==None: eps=0.005
    if steps==None: steps=10
    N = s.size
    S = np.repeat(s[None,:], N, axis=0)
    Z = np.floor(np.abs(S-S.T)/eps)
    Z[Z>steps] = steps

    return Z
#######################################################################################################################


def wavelet_fn(signal,wavelet_name,samples,rm_coefficients):
  level_w=math.floor(math.log2(samples))
  coeffs = pywt.wavedec(signal, wavelet_name, level=level_w)
  coeffs2 = pywt.wavedec(signal, wavelet_name, level=level_w)
  #rm_coefficients=[1,6,7,8,9,10]  # 1 is higher frequncy

  for i in rm_coefficients:
    coeffs[level_w-i+1][0:]=0
 
  out_data = pywt.waverec(coeffs,wavelet_name ) 
  
  return out_data,coeffs2,coeffs
#######################################################################################################################

def filter_fn(data,window_len):
  data2 = (np.append(data[1:], 0) - data)
  data3 = data2/(np.max(abs(data2)))
  data33 = abs(data3)
  data4 = data33**2
  data33[0:]=data33[0:]+0.001
  shanoon_en = -(data33**2) * np.log10(data33**2)
  shanoon_en_filtered = np.insert(running_mean(shanoon_en, window_len), 0, [0] * (window_len - 1))


  return shanoon_en,shanoon_en_filtered

#######################################################################################################################



def signal_process(signal, samples ,wavelet_name,rm_coefficients,window_len\
                   ,recurrence_eps): 
  data,coeffs2,coeffs=wavelet_fn(signal,wavelet_name,samples,rm_coefficients\
                                 =rm_coefficients)
 
  data5,data6=filter_fn(data,window_len)
  rec=recurrence_fn(data6,recurrence_eps)

  rr, _ = find_peaks(data6, distance=40, height=0.02)

  return signal,data,coeffs2,coeffs,data5,data6,rr,rec


#######################################################################################################################
def indextoaddres(indexc,samples,overlap):
  frame_step=samples*(1-overlap)
  flag=1
  i=0
  frame_counter=0
  while flag:
    frame_counter=(len(mylist[i][1])-samples)/frame_step
    if frame_counter>=(indexc):
      flag=0
    else:
      indexc=indexc-frame_counter-1
      i=i+1 
  m=indexc
  h=indexc%2
  return int(i),int(m),int(h)


#######################################################################################################################
def index_gen(state):
  PATH = "/content/finalindex.pt"
  checkpoint = torch.load(PATH)
  addreses1=checkpoint['labels1'] 
  addreses2=checkpoint['labels2'] 
  addreses3=checkpoint['labels3'] 
  addreses1_1=addreses1[0:len(addreses1)-1]
  addreses2_1=addreses2[0:len(addreses2)-1] 
  addreses3_1=addreses3[0:len(addreses3)-1] 
  if state==1:
     addreses=addreses1+addreses2+addreses3
     random.shuffle(addreses)
     return  addreses , 512
  elif state==2:
    addreses=addreses1_1+addreses2_1+addreses3_1
    random.shuffle(addreses)
    return  addreses , 1024

#######################################################################################################################
def check_nan(data):
  for i in range(100):
    if (max(data[i].tolist())==max(data[i].tolist())):
      pass
    else:
      return 1 

#######################################################################################################################


import pickle
PATH = "datasets/finalindex.pt"

def main_fn(samples,wavelet_type,rm_coefficients,window_len,recurrence_eps\
            ,overlap,save_mode,image_size):#overlap=0.25,0.5,
  global rec_data_seg_ch1,rec_data_seg_ch2,data_seg_ch1,data_seg_ch2,total_data\
         ,rec_total_data,labels
  global display_data
  data_preprocess()
  rec_total_seg=[]
  data_seg_ch1=[]
  total_data=[]   #total_data[i][j] i=cases j=sampeles slide
  rec_total_data=[]
  labels=[]
  lable=0
  frame_counter=0
  counter=0
  flag=0
  
  frame_step=samples*(1-overlap)
  compose = transforms.Compose([transforms.Resize((image_size,image_size))])
  cx=int(1)
  cofile=0
  

  addreses,sam=index_gen(2)
  lendata=len(addreses)  
  print(lendata)
  for index in addreses:#len(mylist)):i in range(100):#
     cofile=cofile+1
     frame_step=samples*(1-overlap)
     i,ii,iii=indextoaddres(index[0],samples,overlap)
     x,x,x,x,x,x,x,rec1=signal_process(mylist[i][index[2]][ii*int(frame_step)\
                                      :(ii)*int(frame_step)+sam],sam,wavelet_type,\
                                       rm_coefficients,window_len,recurrence_eps)
     if check_nan(rec1) :  #check if nan is here
       pass
     else:
       a=torch.from_numpy(np.array([rec1])) 
       a2 = torch.unsqueeze(a, dim =0)
       #a2=xtransform(a)
       a3 = compose(a2)
       labels.append(index[1])
       a3=a3.numpy() 
       total_data.append(a3[0])  
   

     if cofile==int(lendata/6) :
       
       cofile=0
       print(cx)
       if cx != 6:
         a = {'len':int(len(total_data)) ,'batch_label': 'training batch ' \
              +str(cx)+' of 5', 'labels': labels, 'data': np.array([total_data])}
         with open('datasets/ECG3class_1ch/data_batch_'+str(cx), 'wb') as f:
           pickle.dump(a, f)
       if cx==6 :
         b={'len':int(len(total_data)),'batch_label': 'test batch 1 of 1',\
            'labels': labels, 'data': np.array([total_data])}
         with open('datasets/ECG3class_1ch/test_batch', 'wb') as f:
           pickle.dump(b, f)
         break         
       labels=[]
       cx=cx+1
       total_data=[]


  

  return 


 

if __name__=='__main__':
  main_fn(samples=512,wavelet_type="db3",rm_coefficients=[1,6,7,8,9,10],window_len=10,recurrence_eps=0.001,overlap=0,save_mode="PAF",image_size=100)




#######################################################################################################################









