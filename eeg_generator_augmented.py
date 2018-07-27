#eeg_generator_fast.py
import numpy as np
import pandas as pd
import keras
import pickle
from sklearn.model_selection import train_test_split

'''
This file holds the generator for eeg neural net training which
augments the eeg data with the distance of the sample from the 
multivariate gaussian distributions for each of the events
''' 

'''
The distance used can be either mahalanobis distance or (this is not technically distance,
but it serves a similar function) the probability under a gaussian mixture model, ignoring the 
constant factor of 1 over root two pi.  The constant factor is likely irrelevant to the neural net
since it essentially rescales all the data with the weights
'''

'''
This class supports unaugmented training, so it is the only one used in producing any final
versions of trained networks
'''


class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size=30, dim1=32, n_evals=16,
        target_dim=6, steps_back=50, shuffle=False, mode='event', dist='maha'):
        '''Initialization'''
        self.mode = mode
        self.batch_size = batch_size
        self.steps_back = steps_back 
        self.target_dim = target_dim
        self.file_IDs = pickle.load(open("file_IDs", "rb"))
        self.file_lens = pickle.load(open("file_lengths", "rb"))
        self.data = pickle.load(open("eeg_train_data_arrs", "rb"))
        self.labels = pickle.load(open("eeg_train_label_arrs", "rb"))
        if mode!='unaugmented':
            if mode=='event':
                self.evecs = np.load('eeg_event_inv_evecs.npy')
                self.evals = np.load('eeg_event_inv_evals.npy')
                self.means = np.load('eeg_event_means.npy')
            elif mode=='sub_event':
                self.evecs = np.load('eeg_sub_event_inv_evecs.npy')
                self.evecs = np.concatenate((self.evecs[:,0,:,:],self.evecs[:,1,:,:]))
                self.evals = np.load('eeg_sub_event_inv_evals.npy')
                self.evals = np.concatenate((self.evals[:,0,:],self.evals[:,1,:]))
                self.means = np.load('eeg_sub_event_means.npy')
                self.means = np.concatenate((self.means[:,0,:],self.means[:,1,:]))

            else:
                raise ValueError("Unrecognized mode for covariance matrices")

            if dist=='gmm':
                if mode=='event':
                    self.norms = np.load('eeg_event_evals.npy')
                    self.norms = self.norms[:,:-n_evals]
                    self.norms = np.prod(self.norms, axis=1)
                elif mode=='sub_event':
                    self.norms = np.load('eeg_sub_event_evals.npy')
                    self.norms = np.concatenate((self.norms[:,0,:],self.norms[:,1,:]))
                    self.norms = self.norms[:,:-n_evals]
                    self.norms = np.prod(self.norms, axis=1)
            self.evecs = self.evecs[:,:,:-n_evals]
            self.evals = self.evals[:,:-n_evals]
            self.evals = np.array([np.diag(e) for e in self.evals])
            self.num_centers = self.evals.shape[0]
            #if the input is being augmented, dimension is larger
            self.dim = (steps_back, dim1+self.num_centers)
        else:
            self.dim = (steps_back, dim1)
        self.fill_list_IDs()
        self.dist = dist
        self.shuffle = shuffle
        self.on_epoch_end()


    def get_num_classes(self):
        return self.target_dim

    
    def get_input_shape(self):
        '''
        returns dimension, which is useful since dimensions depend upon mode,
        and the input shape is needed for constructing the model
        '''
        return self.dim

    
    def fill_list_IDs(self):
        #custom data type of name and index together for storing list IDs
        #as a combination of a file and index within that file
        dt = np.dtype([('name', np.unicode_, 35), ('index', np.uint32)])
        self.list_IDs = np.empty(self.num_samples(), dtype=dt)

        ind = 0
        for ID in self.file_IDs:
            for i in np.arange(self.steps_back, self.file_lens[ID]):
                self.list_IDs[ind] = (ID, i)
                ind += 1
        #test train split
        self.list_IDs, self.test_IDs = train_test_split(self.list_IDs, test_size=0.2)
        print("list_IDs and test_IDs filled")


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs) // self.batch_size

    
    def num_samples(self):
        'Denotes the number of samples in the data before test/train split'
        total = 0
        for item in self.file_lens.items():
            total += item[1] - self.steps_back
        return total

    
    def add_distances(self, feat):
        dists = np.empty(self.num_centers, dtype=np.float32)
        for j in range(self.num_centers):
            diff = feat-self.means[j]
            proj = diff.T @ self.evecs[j]
            dist = (proj @ self.evals[j] @ proj.T)**0.5
            if self.dist =='maha':
                dists[j] = dist
            else:
                dists[j] = np.exp(dist)/(self.norms[j]**0.5)
        return np.hstack((dists, feat)).astype(np.float32)

    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # Initialization

        X = np.empty((self.batch_size, *self.dim), 
            dtype=np.float32)
        y = np.empty((self.batch_size, self.target_dim), 
            dtype=np.int16)

        # Generate data
        for i, (ID, ind) in enumerate(list_IDs_temp):
        # Store sample
            
            eeg_feats = self.data[ID][ind-self.steps_back:ind,:]
            if self.mode!='unaugmented':
                feats = [self.add_distances(feat) for feat in eeg_feats]
            else:
                feats = eeg_feats

            X[i,:] = np.array(feats)
            
            temp = self.labels[ID][ind-1,:]
           
            y[i,:] = self.labels[ID][ind-1,:]

        return X, y
    
    
    def test_generation(self, num_epochs):
        '''
        Generate the test data set from the IDs witheld from training
        '''
        xsamples=[]
        ysamples=[]
        np.random.shuffle(self.test_IDs)
        for i in range(num_epochs):
        
            indexes = self.test_IDs[i*self.batch_size:(i+1)*self.batch_size]
            x,y = self.__data_generation(indexes)
            xsamples.append(x)
            ysamples.append(y)
        X = np.concatenate(xsamples)
        y = np.concatenate(ysamples)
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
