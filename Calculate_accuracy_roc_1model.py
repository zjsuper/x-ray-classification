# -*- coding: utf-8 -*-
"""
Created on Mon Dec 9 18:57:18 2017

@author: Sicheng Zhou
"""
# Import libraries
import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn.model_selection as le_ms 
import sklearn.preprocessing as le_pr 
import sklearn.linear_model as le_lm 
import sklearn.metrics as le_me


# extract features and labels from the .h5 file
X_train = []
X_test = []

#models "gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"
models = ["gap_InceptionV3.h5"]

for filename in models:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])
        y_test = np.array(h['test_label'])
        
# transfer to np array
X_train = np.concatenate(X_train, axis=1) 
X_test = np.concatenate(X_test, axis=1)
print(X_train.shape)  
print(X_test.shape) 
        
# shuffle the training set
np.random.seed(2017)
X_train, y_train = shuffle(X_train, y_train)

# scale features with their median and interquantile range
scaler = le_pr.RobustScaler(with_centering = True, with_scaling = True, 
                                quantile_range = (25.0, 75.0))
scaler.fit(X_train)
trainX = scaler.transform(X_train)
testX = scaler.transform(X_test)

#trainX = X_train.copy()
#testX = X_test.copy()

# directly dropout and classification
input_tensor = Input(trainX.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)
model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(trainX, y_train, batch_size=128, nb_epoch=20, validation_split=0.1)
y_pred = model.predict(testX, verbose=1)


# Calculate fpr,tpr,auc
false_positive_rate, true_positive_rate, thresholds = le_me.roc_curve(y_test, y_pred)
roc_auc = le_me.auc(false_positive_rate, true_positive_rate)

#set threshold
thresh = 0.988

#transform probabilities to class
y_predict = y_pred.copy()
y_predict[y_predict > thresh] = int(1)
y_predict[y_predict <= thresh] = int(0)

# calculate accuracy
accurate = le_me.accuracy_score(y_test, y_predict, normalize=True, sample_weight=None)
print('Accuracy:',accurate)

# get classification report
cl_re = le_me.classification_report(y_test, y_predict)
print(cl_re)

# calculate confusion matrix
con_matrix = le_me.confusion_matrix(y_test, y_predict, labels=None, sample_weight=None)
print('Confusion Matrix:')
print(con_matrix)


# same pipeline but not scale the features
input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)
model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, nb_epoch=20, validation_split=0.1)
y_pred_noscale = model.predict(X_test, verbose=1)

#print(y_pred,y_test)

# Calculate fpr,tpr,auc
false_positive_rate_noscale, true_positive_rate_noscale, thresholds_noscale = le_me.roc_curve(y_test, y_pred_noscale)
roc_auc_noscale = le_me.auc(false_positive_rate_noscale, true_positive_rate_noscale)

#set threshold
thresh = 0.985

#transform probabilities to class
y_predict_noscale = y_pred_noscale.copy()
y_predict_noscale[y_predict_noscale > thresh] = int(1)
y_predict_noscale[y_predict_noscale <= thresh] = int(0)

# calculate accuracy
accurate = le_me.accuracy_score(y_test, y_predict_noscale, normalize=True, sample_weight=None)
print('Accuracy:',accurate)

# get classification report
cl_re = le_me.classification_report(y_test, y_predict_noscale)
print(cl_re)

# calculate confusion matrix
con_matrix = le_me.confusion_matrix(y_test, y_predict_noscale, labels=None, sample_weight=None)
print('Confusion Matrix:')
print(con_matrix)

# roc curves for both scaled and no scaled features
plt.figure(figsize = (6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate_noscale, true_positive_rate_noscale, 'b',label='Non-scaled features AUC = %0.2f'% roc_auc_noscale,color="blue")
plt.plot(false_positive_rate, true_positive_rate, 'b',label='Scaled features AUC = %0.2f'% roc_auc,color="green")
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--',label='Random Guessing Line',color="red")
plt.legend(loc='lower right')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show()




























