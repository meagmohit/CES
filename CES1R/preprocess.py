import sys
sys.path.append('/g/g91/agarwal3/mne-python/')


import mne
import os
import numpy as np
import random
import pickle
  
batch_size = 16
step_size = 512
num_electrodes = 14
num_class = 2
data_path = 'data/'


train_files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(os.path.join(data_path, "train")) if f[-1]=='t']
test_files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(os.path.join(data_path, "test")) if f[-1]=='t']

train_total = len(train_files)
test_total = len(test_files)

data = np.zeros([train_total, num_electrodes, step_size])
label = np.zeros([train_total, 1])
for ind in xrange(train_total):
  filename = train_files[ind]
  filepath_data = os.path.join('data/','train',filename +'.txt')
  filepath_lab = os.path.join('data/','train',filename +'.lab')
  data[ind,:,:] = np.transpose(np.loadtxt(filepath_data))
  f_temp = open(filepath_lab,'r')
  lab_temp = f_temp.read()
  f_temp.close()
  if lab_temp=='Like':
    label[ind,0]=1
  elif lab_temp=='Disike':
    label[ind,0]=2
  else:
    print("\033[91m  Error in Labels check .. \033[0m")
csp = mne.decoding.CSP(n_components=5, cov_est='epoch', transform_into='csp_space')
mycsp = csp.fit(data,label)


file_pi = open('csp.obj', 'w') 
#pickle.dump(mycsp, file_pi) 
     
def next_batch(self, sess, flag):
  if flag=='train':
    total_files = train_total
    batch_index = random.sample(xrange(total_files),batch_size)
    batch_files =  [train_files[i] for i in batch_index]
    mode_str = 'train'
    
  data = np.zeros([batch_size, step_size, 5])
  label = np.zeros([batch_size, num_class])
    
  for ind, filename in enumerate(batch_files):
    filepath_data = os.path.join('data/',mode_str,filename +'.txt')
    filepath_lab = os.path.join('data/',mode_str,filename +'.lab')
    X = (np.loadtxt(filepath_data))
    rdata = (mycsp.transform(X))
      # Data Pre-Processing
      # Normalization
    data[ind, :, :] = np.multiply(rdata - (rdata.mean(0).reshape([1,5])), 1/(10e-12 + rdata.std(0).reshape([1,5])))
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

