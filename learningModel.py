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
    
    train_size = 0.80
    dev_size = 0.20
    
    def __init__(self, X_train_nn, X_test_nn, y_train_dev, y_test, layers, variables, labels):
        
        #if dataset == ' Standard'
        
        self.areaIdx  = variables.index('Area')
        self.labelIdx = variables.index(labels)
        
        self.layers = layers
        
        self.nFeatures, self.nSamples = X_train_nn.shape
        
        self.X_mean = np.mean(X_train_nn, axis = 1, keepdims = True)
        self.X_std  = np.std(X_train_nn,  axis = 1, keepdims = True, ddof = 1)+1e-10
        
        self.X_mean[self.areaIdx] = 0.0
        self.X_std[self.areaIdx]  = 1.0
        
        X_train_dev = (X_train_nn - self.X_mean)/self.X_std
        self.X_test = (X_test_nn  - self.X_mean)/self.X_std
        self.y_test = y_test
        
        #print(X_train_dev.shape, X_test.shape, y_train_dev.shape, y_test.shape)
        
        self.X_train, self.X_dev, self.y_train, self.y_dev = self.train_dev_split(X_train_dev.T, y_train_dev.T)
        
        train_x = tf.data.Dataset.from_tensor_slices(self.X_train.T)
        train_y = tf.data.Dataset.from_tensor_slices(self.y_train.T)

        dev_x = tf.data.Dataset.from_tensor_slices(self.X_dev.T)
        dev_y = tf.data.Dataset.from_tensor_slices(self.y_dev.T)

        test_x = tf.data.Dataset.from_tensor_slices(self.X_test.T)
        test_y = tf.data.Dataset.from_tensor_slices(self.y_test.T)

        ### CONSTANTS DEFINING THE MODEL ####
        
        print('=====================================================')
        print('Train set X shape: ' + str(self.X_train.shape))
        print('Train set y shape: ' + str(self.y_train.shape))
        print('Dev   set x shape: ' + str(self.X_dev.shape))
        print('Dev   set y shape: ' + str(self.y_dev.shape))
        print('X: ' + str(type(self.X_train)) + ', y: ' + str(type(self.y_train)))
        print('Layer dimensions: ' + str(layers))
        print('=====================================================\n')

        self.parameters, costs = gatti.model(train_x, train_y, dev_x, dev_y, test_x, test_y, self.layers, self.areaIdx,
                                        learning_rate = 0.001, num_epochs = 201, minibatch_size = 128)
        
        self.predictionsRMSE()
    
    def predictionsRMSE(self, test = False):
    
        if test == True:
            X_pred = np.vstack((self.X_train.T, self.X_dev.T, self.X_train.T)).T
            HF_Cp  = np.vstack((self.y_train.T, self.y_dev.T, self.y_train.T)).T
        else:
            X_pred = np.vstack((self.X_train.T, self.X_dev.T)).T
            HF_Cp  = np.vstack((self.y_train.T, self.y_dev.T)).T
        
        nFeatures, nPoints = X_pred.shape
        
        pred_X = tf.data.Dataset.from_tensor_slices(X_pred.T)
        Cp_HF  = tf.data.Dataset.from_tensor_slices(HF_Cp.T)
    
        pred_dataset = tf.data.Dataset.zip((pred_X, Cp_HF))
        pred_bacth = pred_dataset.batch(nPoints).prefetch(8)
        
        for (X, y) in pred_bacth:
            
            # Compute MF error (de-normalize data + RMSE computation)
            _, y_hat = gatti.forward_propagation(tf.transpose(X), self.parameters, self.layers)
            NN_RMSE  = gatti.compute_cost(tf.transpose(y), y_hat, self.layers, tf.gather(X, self.areaIdx, axis=1))
            
            # Compute LF error (de-normalize data + RMSE computation)
            y_LF = tf.gather(X, self.labelIdx, axis=1)*self.X_std[self.labelIdx] + self.X_mean[self.labelIdx]
            LF_RMSE  = gatti.compute_cost(tf.transpose(y), y_LF, self.layers, tf.gather(X, self.areaIdx, axis=1))
        
        print('=====================================================')
        print('RMSE comparison MF Neural Net vs LF LES over test_dev')
        print("MF NN  integral RMSE %f" %NN_RMSE)
        print("LF LES integral RMSE %f" %LF_RMSE)
        print('=====================================================')
        
        return


    def train_dev_split(self, X, y):
        
        if abs(self.train_size + self.dev_size - 1.0) > 1e-6:
            raise Exception("Summation of dataset splits should be 1")
        
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=self.dev_size, random_state=42)
        
        return X_train.T, X_dev.T, y_train.T, y_dev.T
        
        
        
class preprocess_features:

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
            
            """
            Train set: 
                - data from LF model @ HF angles
                - label from HF model @ HF angles
            """
            trainDF     = self.read_file([self.resolutions['LF']], self.angles['HF'])
            trainLabels = self.read_file([self.resolutions['HF']], self.angles['HF'])
            
            
            """
            Test set: 
                - data from LF model @ withheld (test) angles
                - label from HF model @ withheld (test) angles
            """
            testAngles = np.sort(list(set(self.angles['LF']) - set(self.angles['HF'])))
            testDF     = self.read_file([self.resolutions['LF']], testAngles)
            testLabels = self.read_file([self.resolutions['HF']], testAngles)

            X_train = trainDF[variables].values.T
            y_train = (trainLabels[labels].values).reshape(1,-1)

            X_test = testDF[variables].values.T
            y_test = (testLabels[labels].values).reshape(1,-1)
            
        
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

angles     = {'LF': [0,10,20,30,40,50,60,70,80,90], 'HF': [0,20,40,60,80]}
resolution = {'LF': 'Coarsest', 'HF': 'Coarse'}
variables  = ['CfMean','TKE','U','gradP','theta','meanCp','rmsCp','peakminCp','peakMaxCp','Area']
labels     = 'meanCp'

datasplit = preprocess_features(angles, resolution, variables, labels, 'MultiFidelity')

X_train_dev, X_test, y_train_dev, y_test = datasplit.split_dataset()
        
layers = [X_train_dev.shape[0],10,10,5,3,1]

LR = neural_networks(X_train_dev, X_test, y_train_dev, y_test, layers, variables, labels)
