# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:41:34 2017

@author: zhou sicheng
"""

import pandas as pd
import shutil
import os

# read csv
df = pd.read_csv('datainfo.csv',header = 0)

# extract filename and cardiomegaly information
dfles = df[['FileName','Cardiomegaly']]
print(dfles)

# create empty name list
list_0 = []
list_1 = []

for index, row in dfles.iterrows():
    if row['Cardiomegaly'] == 0:
        list_0.append(row['FileName'])
    if row['Cardiomegaly'] == 1:
        list_1.append(row['FileName'])   

print(len(list_1))

source = './test/'
dest0 = './test_0'
dest1 = './test_1'

files = os.listdir(source)

for f in files:
    if f in list_0:
        shutil.move(source+f, dest0)
    if f in list_1:
        shutil.move(source+f, dest1)
        
source2 = './train/'
dest20 = './train_0'
dest21 = './train_1'

files_train = os.listdir(source2)
for f in files_train:
    if f in list_0:
        shutil.move(source2+f, dest20)
    if f in list_1:
        shutil.move(source2+f, dest21)