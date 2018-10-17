# Created by Mohit Agarwal on 10/17/2018
# Georgia Institute of Technology
# Copyright 2018 Mohit Agarwal. All rights reserved.

# Processing and classification for consumer choice classification using xDAWN and Riemannian Framework

import numpy as np
import classif as clf
from sklearn.preprocessing import Normalizer
from sklearn import svm
import pyriemann

filters_xdawn = 5
subelec_xdawn = [9,10,11,12,14] #range(0,19,1) #
filters_elecselect = 2
nelec_elecselect = 2

X_train = np.load('processed_data/train_data.npy')
Y_train = np.load('processed_data/train_label.npy')
X_test = np.load('processed_data/test_data.npy')
Y_test = np.load('processed_data/test_label.npy')

X_train = X_train - np.mean(X_train,axis=2,keepdims=True)
X_test = X_test - np.mean(X_test,axis=2,keepdims=True)

Y_train = np.reshape(Y_train,[Y_train.shape[0]])
Y_test = np.reshape(Y_test,[Y_test.shape[0]])

filt_1 = clf.XdawnCovariances(nfilter=filters_xdawn, subelec=subelec_xdawn)
X_train = filt_1.fit_transform(X_train, Y_train)

#filt_2 = clf.ElectrodeSelect(nfilters=filters_elecselect, nelec=nelec_elecselect, metric='riemann')
#X_train = filt_2.fit_transform(X_train, Y_train)

#filt_3 = clf.TangentSpace(metric='logeuclid', tsupdate = False)
#X_train = filt_3.fit_transform(X_train, Y_train)

#filt_4 = Normalizer(norm='l1')
#X_train = filt_4.fit_transform(X_train, Y_train)
#filt_5 = ElasticNet(l1_ratio=0.05, alpha=0.02, normalize=True)

#filt_5 = svm.SVC(kernel='rbf', C=0.001, class_weight='balanced')
filt_5 = pyriemann.classification.MDM(metric='riemann',n_jobs=1)
filt_5.fit(X_train,Y_train)
score_train = filt_5.predict(X_train)

X_test = filt_1.transform(X_test)
#X_test = filt_2.transform(X_test)
#X_test = filt_3.transform(X_test)
#X_test = filt_4.transform(X_test)
score_test = filt_5.predict(X_test)

print np.mean(score_train==Y_train)
print np.mean(score_test==Y_test)
