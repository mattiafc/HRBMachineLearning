import numpy   as np
import pandas  as pd
import tensorflow as tf
import tfNeuralNetGATTI as gatti
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt

class neural_networks:

    X_mean = 0
    X_std  = 1
    
    y_mean = 0
    y_std  = 1    
    
    def __init__(self, X_train_nn, X_test_nn, y_train_nn, y_test_nn, layers):
        
        #if dataset == ' Standard'
        
        self.nFeatures, self.nSamples = X_train_nn.shape
        
        #self.X_mean = np.mean(X_train_nn, axis = 1, keepdims = True)
        #self.X_std  = np.std(X_train_nn,  axis = 1, keepdims = True, ddof = 1)+1e-10
        
        #X_train = (X_train_nn - self.X_mean)/self.X_std
        #X_test  = (X_test_nn  - self.X_mean)/self.X_std
        
        y_train = y_train_nn
        y_test  = y_test_nn
        
        train_x = tf.data.Dataset.from_tensor_slices(X_train.T)

        test_x = tf.data.Dataset.from_tensor_slices(X_test.T)

        train_y = tf.data.Dataset.from_tensor_slices(y_train.T)

        test_y = tf.data.Dataset.from_tensor_slices(y_test.T)

        ### CONSTANTS DEFINING THE MODEL ####
        
        print('=====================================================')
        print('Train set X shape: ' + str(X_train.shape))
        print('Test  set y shape: ' + str(y_train.shape))
        print('Train set x shape: ' + str(X_test.shape))
        print('Test  set y shape: ' + str(y_test.shape))
        print('X: ' + str(type(X_train)) + ', y: ' + str(type(y_train)))
        print('Layer dimensions: ' + str(layers))
        print('=====================================================\n')

        parameters, costs = gatti.model(train_x, train_y, test_x, test_y, layers)
        
        #parameters, costs = self.L_layer_model(self.X_train, self.y_train)
        
        #pred_train = predict(self.X_train, self.y_train, parameters)
        #accuracy_train = np.sum(pred_train == self.y_train)/self.y_train.shape[1]
        
        #pred_test = predict(self.X_test, self.y_test, parameters)
        #accuracy_test = np.sum(pred_test == self.y_test)/self.y_test.shape[1]
        
        #print('=====================================================')
        #print('Train set accuracy is: ' + str(accuracy_train))
        #print('Test  set accuracy is: ' + str(accuracy_test))
        #print('=====================================================\n')
        
        

class preprocess_features:
    
    train_size = 0.80
    test_size = 0.20

    def __init__(self, angles, resolutions, features, label, dataset = 'Standard'):

        self.angles      = angles
        self.resolutions = resolutions
        self.features    = features
        self.label       = label
        self.dataset     = dataset

    def split_dataset(self):
        
        if self.dataset == 'Standard':
            
            DF = self.read_file(self.resolutions, self.angles)

            X = DF[variables].values.T
            y = (DF[labels].values).reshape(1,-1)
            
            X_train, X_test, y_train, y_test = self.ordinary_train_test(X.T, y.T)
        
        elif self.dataset == 'MultiFidelity':
            
            LFTrainDF = self.read_file([self.resolutions['LF']], self.angles['LF'])
            HFTrainDF = self.read_file([self.resolutions['HF']], self.angles['HF'])
            
            trainDF = pd.concat([LFTrainDF, HFTrainDF], axis=0)
            print(trainDF)
            testAngles = np.sort(list(set(self.angles['LF']) - set(self.angles['HF'])))
            testDF = self.read_file([self.resolutions['HF']], testAngles)
            print(testDF)

            X_train = trainDF[variables].values.T
            y_train = (trainDF[labels].values).reshape(1,-1)

            X_test = testDF[variables].values.T
            y_test = (testDF[labels].values).reshape(1,-1)
            
        
        print('=====================================================')
        print('Total number of samples is: ' + str(X_train.shape[1]+X_test.shape[1]))
        print('Training set is ' + str(X_train.shape[1]) + ' samples')
        print('Test     set is ' +str(X_test.shape[1])  + ' samples')
        print('=====================================================\n')

        return X_train, X_test, y_train, y_test

    def ordinary_train_test(self, X, y):
        
        if abs(self.train_size + self.test_size - 1.0) > 1e-6:
            raise Exception("Summation of dataset splits should be 1")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        
        return X_train.T, X_test.T, y_train.T, y_test.T
    
    
    def read_file(self, resolutions, angles):
            
        data = []
        
        for res in resolutions:
        
            for ang in angles:
                
                data.append(pd.read_csv(str('Features/' + res + str(ang))))
        
        return pd.concat(data, axis=0) 


np.random.seed(3)

#angles     = [0,10,20,30,40,50,60,70,80,90]
#resolution = ['Coarsest','Coarse']
#variables  = ['CfMean','TKE','U','gradP','rmsCp','peakminCp','peakMaxCp','theta','LV0','Area']
#labels     = 'meanCp'

angles     = {'LF': [0,10,20,30,40,50,60,70,80,90], 'HF': [0,30,60,90]}
resolution = {'LF': 'Coarsest', 'HF': 'Coarse'}
variables  = ['CfMean','TKE','U','gradP','meanCp','peakminCp','peakMaxCp','theta','LV0','Area']
labels     = 'rmsCp'

datasplit = preprocess_features(angles, resolution, variables, labels, dataset)

datasplit = preprocess_features(angles, resolution, variables, labels, 'MultiFidelity')

X_train, X_test, y_train, y_test = datasplit.split_dataset()
        
layers = [X_train.shape[0],5,5,1]
#layers = [n_x,10,7,6,5,4,1]

LR = neural_networks(X_train, X_test, y_train, y_test, layers)
