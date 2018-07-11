#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 18:41:54 2018

@author: mshokry

https://www.drivendata.org/competitions/2/warm-up-predict-blood-donations/page/6/

### BEST MODEL ###
Score 
0.4429

https://www.kaggle.com/pcharambira/predicting-blood-donations
https://github.com/hbergen/predict-blood-donations/blob/master/Predicting-Blood-Donations-for-DrivenData.ipynb
http://matthewalanham.com/Students/2017_MWDSI_Deepti.pdf


"""
from keras.models import Sequential
from keras.layers.core import Activation, Dropout,Dense
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,log_loss
import keras.losses as losses
import pandas as pd
import numpy as np
import matplotlib.pylab as plt 

#Reading Data
data = pd.read_csv('train.csv')

col_names = {'Unnamed: 0': 'id', \
                     'Months since Last Donation': 'months_since_last', \
                     'Number of Donations': 'n_donations', \
                     'Total Volume Donated (c.c.)': 'total_volume', \
                     'Months since First Donation': 'months_since_first', \
                     'Made Donation in March 2007': 'donate'}
all_features = ['months_since_last', 'n_donations', 'total_volume',
       'months_since_first', 'months_donated','per_moth']
features = ['months_since_last', 'n_donations', 'total_volume',
       'months_donated','months_since_first',]
target = ['donate']
data.rename(columns=col_names,inplace=True)
#New Feature
data['months_donated'] = data['months_since_first'] - data['months_since_last']
#data['per_moth'] = data['total_volume'] / data['months_donated'] if data['months_donated'] != 0  else 0.0

X = data.loc[:,features].values
y = data.loc[:,target].values

#Testing Data for submittion
data_t = pd.read_csv('test.csv')
data_t.rename(columns=col_names,inplace=True)
data_t['months_donated'] = data_t['months_since_first'] - data_t['months_since_last']
#data_t['per_moth'] = data_t['total_volume'] / data_t['months_donated']
X_T = data_t.loc[:,features].values
X_index = data_t.loc[:,'id'].values

#Some Drawings
plt.style.use("seaborn-white") 
#plt.(data['months_donated'], data['donate'].astype(bool)).plot(kind='bar')


#Scaling Data
scaler = StandardScaler() 
scaler.fit(X)
X = scaler.transform(X)

X_T = scaler.transform(X_T)
#Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5000)

#KEras
checkpointer = ModelCheckpoint(filepath='weights.best.hdf5', verbose=1, save_best_only=True)

model = Sequential()
model.add(Dense(output_dim=512,input_dim=X_train.shape[1] , init='uniform'))
model.add(Activation("relu"))
model.add(Dropout(0.01))
model.add(Dense(output_dim=1024,  init="lecun_uniform"))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(Dense(output_dim=512,  init="lecun_uniform"))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(output_dim=1, input_dim=50, init = "normal"))
model.add(Activation("sigmoid"))

model.compile(loss=losses.binary_crossentropy,optimizer='rmsprop', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train,  epochs=30,
          batch_size=8, verbose=2,validation_split=0.2,callbacks=[checkpointer])

model.load_weights('weights.best.hdf5')
proba = model.predict_proba(X_test)
y_hat =model.predict_classes(X_test)
model.evaluate(X_test,y_test)

print(log_loss(y_test,proba[:,0]))
conv = pd.DataFrame(confusion_matrix(y_test,y_hat))
print(conv)
print(classification_report(y_test,y_hat))
