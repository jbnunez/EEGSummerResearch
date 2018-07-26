#eeg_data_generator.py
import numpy as np
import pandas as pd
import os
import pickle

train_path = 'train/'
dir_list = os.listdir(path=train_path)

#not including self

datafiles = []
labelfiles = []

for file in dir_list:
    if file[-8:] == 'data.csv':
        datafiles.append(train_path+file)
    else:
        labelfiles.append(train_path+file)
        #df = pd.read_csv(train_path+file, index_col=0)
        #print(df.head())
        #print(df.shape)

#5/0
file_starts = [i[:-8] for i in datafiles]

file_lengths = {}
dfs = {}
for file in file_starts:
    df = pd.read_csv(file+'data.csv', index_col=0)
    print(df.head())
    file_lengths[file] = df.shape[0]
    print(df.shape)
    dfs = {}


fileObject = open("file_IDs",'wb') 
pickle.dump(file_starts, fileObject)


filelenObject = open("file_lengths",'wb') 
pickle.dump(file_lengths, filelenObject)


