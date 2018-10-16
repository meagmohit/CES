import numpy as np
import os
import random
import mne
import operator as op
from scipy import stats

Nt = 768	# Number of Sampels
Ne = 19		# Number of Electrodes
mode = 0	# 0: Train/test split for all users, 1: train on different users and test on different
Nw = 50     # Number of waveforms to average
train_split = 0.7
valid_split = 0.15
data_dir = 'data/'

## Small Helpful functions
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom

## Preprocessing and data arrangement functions
def preprocess():
    list_of_data_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and ".dat" in f]
    list_of_labl_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and ".lab" in f]
    total_files = len(list_of_data_files)
    total_subjects = 0
    EEG_data = np.zeros([total_files,Nt,Ne])
    prod_ids = np.zeros([total_files,1])
    sub_ids  = np.zeros([total_files,1])
    index = 0
    for files in list_of_data_files:
        file_data = os.path.join(data_dir,files)
        file_labl = file_data.replace('.dat','.lab')
        X = np.loadtxt(file_data,delimiter=',')         # stores eeg_data
        Y = int(np.loadtxt(file_labl,delimiter=','))  	# stores product_id
        Z = int(files[1:3])                             # stores subject_id
        EEG_data[index,:,:] = X[:,0:19]
        prod_ids[index,0] = Y
        sub_ids[index,0] = Z
        index = index+1
    labels = np.loadtxt(os.path.join(data_dir,'outChoices.txt'))      # subject, trial, item1id, item2id, chosenitem1
    return EEG_data, prod_ids, sub_ids, labels

def train_test_data_mode0(EEG_data, prod_ids, sub_ids, labels):
    total_subjects = int(np.amax(sub_ids))
    total_products = int(np.amax(prod_ids))
    total_prod_in_test = int((1-train_split)*total_products)
    total_prod_in_train = total_products - total_prod_in_test
    total_train_samples = total_subjects*ncr(total_prod_in_train,2)
    total_test_samples = total_subjects*ncr(total_prod_in_test,2)
    X_train = np.zeros([total_train_samples,Ne,Nt])
    Y_train = np.zeros([total_train_samples,1])
    X_test = np.zeros([total_test_samples,Ne,Nt])
    Y_test = np.zeros([total_test_samples,1])
    global_idx_train = 0
    global_idx_test = 0
    for sub_idx in xrange(1,total_subjects+1):
        test_prod_ids = random.sample(range(1,total_products+1),total_prod_in_test)
        train_prod_ids = list(set(range(1,total_products+1)) - set(test_prod_ids))
        train_prod_ids.sort()
        test_prod_ids.sort()
        #X_train,Y_train
        for prod_idx_1 in xrange(0,total_prod_in_train-1):
            for prod_idx_2 in xrange(prod_idx_1+1,total_prod_in_train):
                #Product 1
                t = ((prod_ids==train_prod_ids[prod_idx_1])&(sub_ids==sub_idx))
                EEG_ids = [i for i, x in enumerate(t) if x]                 # Gives the indices where sub_id and prod_id matches
                X_temp = EEG_data[EEG_ids,:,:]                              # X_temp is 50x768x19
                X_temp_avg_1 = np.sum(X_temp,axis=0)                          # X_temp_sum is 768x19  average of all waveforms
                #Product 2
                t = ((prod_ids==train_prod_ids[prod_idx_2])&(sub_ids==sub_idx))
                EEG_ids = [i for i, x in enumerate(t) if x]                 # Gives the indices where sub_id and prod_id matches
                X_temp = EEG_data[EEG_ids,:,:]                              # X_temp is 50x768x19
                X_temp_avg_2 = np.sum(X_temp,axis=0)                        # X_temp_sum is 768x19  average of all waveforms
                #Preparing Data
                X_train[global_idx_train,:,:] = X_temp_avg_1.transpose(1,0) - X_temp_avg_2.transpose(1,0)
                t = (labels[:,0]==sub_idx)&(labels[:,2]==train_prod_ids[prod_idx_1])&(labels[:,3]==train_prod_ids[prod_idx_2])
                label_ids = [i for i, x in enumerate(t) if x]
                Y_train[global_idx_train,:] = int(stats.mode(labels[label_ids,4])[0])
                global_idx_train = global_idx_train+1
                #X_test , Y_test
        for prod_idx_1 in xrange(0,total_prod_in_test-1):
            for prod_idx_2 in xrange(prod_idx_1+1,total_prod_in_test):
                #Product 1
                t = ((prod_ids==test_prod_ids[prod_idx_1])&(sub_ids==sub_idx))
                EEG_ids = [i for i, x in enumerate(t) if x]                 # Gives the indices where sub_id and prod_id matches
                X_temp = EEG_data[EEG_ids,:,:]                              # X_temp is 50x768x19
                X_temp_avg_1 = np.sum(X_temp,axis=0)                          # X_temp_sum is 768x19  average of all waveforms
                #Product 2
                t = ((prod_ids==test_prod_ids[prod_idx_2])&(sub_ids==sub_idx))
                EEG_ids = [i for i, x in enumerate(t) if x]                 # Gives the indices where sub_id and prod_id matches
                X_temp = EEG_data[EEG_ids,:,:]                              # X_temp is 50x768x19
                X_temp_avg_2 = np.sum(X_temp,axis=0)                        # X_temp_sum is 768x19  average of all waveforms
                #Preparing Data
                X_test[global_idx_test,:,:] = X_temp_avg_1.transpose(1,0) - X_temp_avg_2.transpose(1,0)
                t = (labels[:,0]==sub_idx)&(labels[:,2]==test_prod_ids[prod_idx_1])&(labels[:,3]==test_prod_ids[prod_idx_2])
                label_ids = [i for i, x in enumerate(t) if x]
                Y_test[global_idx_test,:] = int(stats.mode(labels[label_ids,4])[0])
                global_idx_test = global_idx_test+1
    #Random Shuffling of Data points
    index_train_shuffle = range(total_train_samples)
    random.shuffle(index_train_shuffle)
    X_train = X_train[index_train_shuffle,:,:]
    Y_train = Y_train[index_train_shuffle,:]
    index_test_shuffle = range(total_test_samples)
    random.shuffle(index_test_shuffle)
    X_test = X_test[index_test_shuffle,:,:]
    Y_test = Y_test[index_test_shuffle,:]
    return X_train,Y_train,X_test,Y_test

EEG_data, prod_ids, sub_ids, labels = preprocess()
X_train,Y_train,X_test,Y_test = train_test_data_mode0(EEG_data, prod_ids, sub_ids, labels)
np.save('processed_data/train_data.npy',X_train)
np.save('processed_data/train_label.npy',Y_train)
np.save('processed_data/test_data.npy',X_test)
np.save('processed_data/test_label.npy',Y_test)
