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
    self._num_features = config.num_features

    self._batch_state = 0

    self._train_files = train_files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(os.path.join(data_path, "train")) if f[-1]=='t']
    self._test_files = test_files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(os.path.join(data_path, "test")) if f[-1]=='t']

    self._train_total = len(train_files)
    self._test_total = len(test_files)

    #data = np.zeros([self._train_total, self._num_electrodes, self._step_size])
    #label = np.zeros([self._train_total, 1])
    #for ind in xrange(self._train_total):
    #  filename = self._train_files[ind]
    #  filepath_data = os.path.join('data/','train',filename +'.txt')
    #  filepath_lab = os.path.join('data/','train',filename +'.lab')
    #  data[ind,:,:] = np.transpose(np.loadtxt(filepath_data))
    #  f_temp = open(filepath_lab,'r')
    #  lab_temp = f_temp.read()
    #  f_temp.close()
    #  if lab_temp=='Like':
    #    label[ind,0]=1
    #  elif lab_temp=='Disike':
    #    label[ind,0]=2
    #  else:
    #    print("\033[91m  Error in Labels check .. \033[0m")
    #csp = mne.decoding.CSP(n_components=5, cov_est='epoch', transform_into='csp_space')
    file_pi = open('csp.obj', 'r') 
    self._csp = mycsp = pickle.load(file_pi) #csp.fit(data,label)
    file_pi.close()
    
     
  def next_batch(self, sess, flag):
    if flag=='train':
      total_files = self._train_total
      batch_index = random.sample(xrange(total_files),self._batch_size)
      batch_files =  [self._train_files[i] for i in batch_index]
      mode_str = 'train'
    elif flag=='test':
      total_files = self._test_total
      batch_index = random.sample(xrange(total_files),self._batch_size)
      batch_files =  [self._test_files[i] for i in batch_index]
      mode_str = 'test'
    else:
      print("\033[91m Incorrect parameter provided for getting next_batch ... \033[0m")
      return

    
    data = np.zeros([self._batch_size, self._step_size, self._num_features])
    label = np.zeros([self._batch_size, self._num_class])
    
    for ind, filename in enumerate(batch_files):
      filepath_data = os.path.join('data/',mode_str,filename +'.txt')
      filepath_lab = os.path.join('data/',mode_str,filename +'.lab')
      X = (np.loadtxt(filepath_data))
      rdata = (self._csp.transform(X))
      # Data Pre-Processing
      # Normalization
      data[ind, :, :] = np.multiply(rdata - (rdata.mean(0).reshape([1,self._num_features])), 1/(10e-12 + rdata.std(0).reshape([1,self._num_features])))
      f_temp = open(filepath_lab,'r')
      lab_temp = f_temp.read()
      f_temp.close()
      if lab_temp=='Like':
        label[ind,0]=1
      elif lab_temp=='Disike':
        label[ind,1]=1
      else:
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
