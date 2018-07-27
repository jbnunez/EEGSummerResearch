#eeg_generator.py
import numpy as np
import pandas as pd
import keras
import pickle


class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size=30, dim1=32, 
        target_dim=6, steps_back=10, shuffle=True):
        'Initialization'
        #dimension is equal to the 400 features from the convolutional encoder
        #plus 200 features from the covariance matrix encoder
        self.dim = (steps_back, dim1)
        self.batch_size = batch_size
        self.steps_back = steps_back 
        self.target_dim = target_dim
        self.file_IDs = pickle.load(open("file_IDs", "rb"))
        self.file_lens = pickle.load(open("file_lengths", "rb"))
        self.fill_list_IDs()

        self.shuffle = shuffle
        self.on_epoch_end()

    def get_num_classes(self):
        return self.target_dim

    def fill_list_IDs(self):
        dt = np.dtype([('name', np.unicode_, 21), ('index', np.uint16)])
        self.list_IDs = np.empty(self.num_samples(), dtype=dt)

        ind = 0
        for ID in self.file_IDs:
            for i in range(self.steps_back, self.file_lens[ID]):
                self.list_IDs[ind] = (ID, i)
                ind += 1



    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_samples() // self.batch_size

    def num_samples(self):
        'Denotes the number of batches per epoch'
        total = 0
        for item in self.file_lens.items():
            total += item[1]
        return total

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim)
        # Initialization

        X = np.empty((self.batch_size, *self.dim), 
            dtype=np.int16)
        y = np.empty((self.batch_size, self.target_dim), 
            dtype=np.uint8)


        # Generate data
        for i, (ID, ind) in enumerate(list_IDs_temp):
        # Store sample
            #list_IDs is an 

            X[i,:] = np.array(pd.read_csv(ID+'data.csv', index_col=0))[ind-self.steps_back:ind,:]
           
            y[i,:] = np.array(pd.read_csv(ID+'events.csv', index_col=0))[ind-1,:]

        #conv_encoded = self.conv_encoder.predict(X=conv_input)


        return X, y


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
