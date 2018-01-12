from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('/g/g91/agarwal3/mne-python/')


import mne
import os
import numpy as np
import tensorflow as tf
import random
import pickle

class input(object):
  
  def __init__(self, data_path, config):
   
    self._batch_size = batch_size = config.batch_size
    self._step_size = step_size = config.step_size
    self._num_electrodes = config.num_electrodes
    self._num_class = config.num_class
    self._data_path = data_path
    self._features = config.features
    self._num_features = len(config.features)

    WA2 = np.loadtxt('WA2.txt',delimiter=',')

    self._batch_state = 0

    self._all_files = all_files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(os.path.join(data_path)) if f[-1]=='t']
    random.shuffle(all_files)
    self._all_files = all_files
    #print(all_files)
    sub_id = int(all_files[0][1:3]) -1
    self._top_5 = np.argsort(WA2[sub_id,:])[:5]
    self._low_5 = np.argsort(WA2[sub_id,:])[5:]
   
    self._files_total = len(all_files)
    self._train_total = int(0.8*self._files_total)
    self._test_total = int(0.2*self._files_total)
    
    self._train_files = train_files = all_files[0:self._train_total]
    self._test_files = test_files = all_files[self._train_total:]

  def next_batch(self, sess, flag):
    if flag=='train':
      total_files = self._train_total
      batch_index = random.sample(xrange(total_files),self._batch_size)
      batch_files =  [self._train_files[i] for i in batch_index]
    elif flag=='test':
      total_files = self._test_total
      batch_index = random.sample(xrange(total_files),self._batch_size)
      batch_files =  [self._test_files[i] for i in batch_index]
    else:
      print("\033[91m Incorrect parameter provided for getting next_batch ... \033[0m")
      return

    
    data = np.zeros([self._batch_size, self._step_size, self._num_features])
    label = np.zeros([self._batch_size, self._num_class])
    
    for ind, filename in enumerate(batch_files):
      filepath_data = os.path.join('data/',filename +'.dat')
      filepath_lab = os.path.join('data/',filename +'.lab')
      X = (np.loadtxt(filepath_data, delimiter=','))
      rdata = X[256:512,self._features]
      #rdata = (self._csp.transform(X))
      # Data Pre-Processing
      # Normalization
      data[ind, :, :] = np.multiply(rdata - (rdata.mean(0).reshape([1,self._num_features])), 1/(10e-12 + rdata.std(0).reshape([1,self._num_features])))
      lab_temp = int(np.loadtxt(filepath_lab,delimiter=',')) -1
      if lab_temp in self._top_5:
        label[ind,0]=1
      elif lab_temp in self._low_5:
        label[ind,1]=1
      else:
        print(lab_temp)
        print("\033[91m  Error in Labels check .. \033[0m")
       
    return data, label

   
  @property
  def train_data(self):
    return self._train_files

  @property
  def test_data(self):
    return self._test_files

  @property
  def batch_state(self):
    return self._batch_state

  @property
  def train_total(self):
    return self._train_total

  @property
  def test_total(self):
    return self._test_total
