#eeg_lstm_augmented.py
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Flatten
from keras.layers.recurrent import LSTM
from keras.models import load_model, Model
import keras
from eeg_generator_augmented import DataGenerator


batch_size = 50
sample_dim = 32
steps_back = 50
target_dim = (6)
num_epochs = 1
n_evals=16
mode='sub_event'


train_generator = DataGenerator(batch_size=batch_size, 
    dim1=sample_dim, target_dim=target_dim, 
    steps_back=steps_back, mode=mode, n_evals=n_evals)
input_shape = train_generator.get_input_shape()

model = Sequential()
model.add(LSTM(1000, input_shape=input_shape))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(target_dim, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='nadam')


model.fit_generator(generator=train_generator, epochs=num_epochs, 
                    steps_per_epoch=10000, workers=4, use_multiprocessing=True)