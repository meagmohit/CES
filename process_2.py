# Created by Mohit Agarwal on 10/17/2018
# Georgia Institute of Technology
# Copyright 2018 Mohit Agarwal. All rights reserved.

# Processing and classification for consumer choice classification using N200 and SVMs

import numpy as np
import classif as clf
from sklearn.preprocessing import Normalizer
from sklearn import svm
import pyriemann

X_train = np.load('processed_data/train_data.npy')
Y_train = np.load('processed_data/train_label.npy')
X_test = np.load('processed_data/test_data.npy')
Y_test = np.load('processed_data/test_label.npy')

X_train = X_train - np.mean(X_train,axis=2,keepdims=True)
X_test = X_test - np.mean(X_test,axis=2,keepdims=True)

Y_train = np.reshape(Y_train,[Y_train.shape[0]])
Y_test = np.reshape(Y_test,[Y_test.shape[0]])

X_train_new = np.zeros([X_train.shape[0],30])
X_test_new = np.zeros([X_test.shape[0],30])

X_train_new[:,0:3] = np.mean(X_train[:,[9,11,14],307:333],axis=2)
#X_train_new[:,3:6] = np.mean(X_train[:,[9,11,14],360:460],axis=2)
#X_train_new[:,6:9] = np.mean(X_train[:,[9,11,14],460:560],axis=2)
#X_train_new[:,9:12] = np.mean(X_train[:,[9,11,14],560:660],axis=2)
#X_train_new[:,12:15] = np.mean(X_train[:,[9,11,14],660:760],axis=2)
X_train_new[:,15:18] = np.mean(X_train[:,[9,11,14],(768+307):(768+333)],axis=2)
#X_train_new[:,18:21] = np.mean(X_train[:,[9,11,14],(768+360):(768+460)],axis=2)
#X_train_new[:,21:24] = np.mean(X_train[:,[9,11,14],(768+460):(768+560)],axis=2)
#X_train_new[:,24:27] = np.mean(X_train[:,[9,11,14],(768+560):(768+660)],axis=2)
#X_train_new[:,27:30] = np.mean(X_train[:,[9,11,14],(768+660):(768+760)],axis=2)


X_test_new[:,0:3] = np.mean(X_test[:,[9,11,14],307:333],axis=2)
#X_test_new[:,3:6] = np.mean(X_test[:,[9,11,14],360:460],axis=2)
#X_test_new[:,6:9] = np.mean(X_test[:,[9,11,14],460:560],axis=2)
#X_test_new[:,9:12] = np.mean(X_test[:,[9,11,14],560:660],axis=2)
#X_test_new[:,12:15] = np.mean(X_test[:,[9,11,14],660:760],axis=2)
X_test_new[:,15:18] = np.mean(X_test[:,[9,11,14],(768+307):(768+333)],axis=2)
#X_test_new[:,18:21] = np.mean(X_test[:,[9,11,14],(768+360):(768+460)],axis=2)
#X_test_new[:,21:24] = np.mean(X_test[:,[9,11,14],(768+460):(768+560)],axis=2)
#X_test_new[:,24:27] = np.mean(X_test[:,[9,11,14],(768+560):(768+660)],axis=2)
#X_test_new[:,27:30] = np.mean(X_test[:,[9,11,14],(768+660):(768+760)],axis=2)


X_train = X_train_new
X_test = X_test_new

filt = svm.SVC(kernel='linear', C=0.4, class_weight='balanced')
filt.fit(X_train,Y_train)
score_train = filt.predict(X_train)

score_test = filt.predict(X_test)
print np.mean(score_train==Y_train)
print np.mean(score_test==Y_test)
