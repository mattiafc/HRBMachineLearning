import numpy   as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from neuralNet import *

import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt

class logistic_regression:

    X_mean = 1
    X_std  = 0
    
    y_mean = 1
    y_std  = 0
    
    layers_dims = [0, 3, 3, 1] 
    
    
    def __init__(self, X_train_nn, X_test_nn, y_train_nn, y_test_nn):
        
        #if dataset == ' Standard'
        
        self.nFeatures, self.nSamples = X_train_nn.shape
        
        self.X_mean = np.mean(X_train_nn, axis = 1, keepdims = True)
        self.X_std  = np.std(X_train_nn, axis = 1, keepdims = True, ddof = 1)+1e-10
        
        self.X_train = (X_train_nn - self.X_mean)/self.X_std
        self.X_test  = (X_test_nn - self.X_mean)/self.X_std
        
        self.y_train = y_train_nn
        self.y_test = y_test_nn
        
        self.layers_dims[0] = X_train.shape[0]
        
        print('=====================================================')
        print('Train set X shape: ' + str(X_train.shape))
        print('Test  set y shape: ' + str(y_train.shape))
        print('Train set x shape: ' + str(X_test.shape))
        print('Test  set y shape: ' + str(y_test.shape))
        print('X: ' + str(type(self.X_train)) + ', y: ' + str(type(self.y_train)))
        print('Layer dimensions: ' + str(self.layers_dims))
        print('=====================================================\n')
        
        parameters, costs = self.L_layer_model(self.X_train, self.y_train)
        
        pred_train = predict(self.X_train, self.y_train, parameters)
        accuracy_train = np.sum(pred_train == self.y_train)/self.y_train.shape[1]
        
        pred_test = predict(self.X_test, self.y_test, parameters)
        accuracy_test = np.sum(pred_test == self.y_test)/self.y_test.shape[1]
        
        print('=====================================================')
        print('Train set accuracy is: ' + str(accuracy_train))
        print('Test  set accuracy is: ' + str(accuracy_test))
        print('=====================================================\n')
        

    # def train_valid_test(self, X, y):
        
    #     if abs(self.train_size + self.test_size + self.valid_size - 1.0) > 1e-6:
    #         raise Exception("Summation of dataset splits should be 1")
        
    #     X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=self.train_size, random_state=42)
        
    #     #X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=self.test_size/(self.valid_size + self.test_size), random_state=42)
        
    #     X_valid = np.zeros((2,2))
    #     y_valid = np.zeros((2,2))
        
    #     return X_train.T, X_valid.T, X_test.T, y_train.T, y_valid.T, y_test.T    

    def L_layer_model(self, X, y, learning_rate = 0.1, num_iterations = 3000, print_cost=True):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []
        
        parameters = initialize_parameters_deep(self.layers_dims)
        
        
        # YOUR CODE ENDS HERE
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
            
            AL, caches = L_model_forward(X, parameters)
            cost = compute_cost(AL, y)
        
            # Backward propagation
            
            grads = L_model_backward(AL, y, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
                    
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
        
        return parameters, costs
        

class preprocess_features:
    
    train_size = 0.60
    test_size = 0.40

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
        
            y[y>-1.5] = 1
            y[y<-1.5] = 0
            
            X_train, X_test, y_train, y_test = self.ordinary_train_test(X.T, y.T)
        
        if self.dataset == 'MultiFidelity':
            
            LFTrainDF = self.read_file([self.resolutions['LF']], self.angles['LF'])
            HFTrainDF = self.read_file([self.resolutions['HF']], self.angles['HF'])
            
            trainDF = pd.concat([LFTrainDF, HFTrainDF], axis=0)
            
            testAngles = np.sort(list(set(self.angles['LF']) - set(self.angles['HF'])))
            
            testDF = self.read_file([self.resolutions['HF']], testAngles)

            X_train = trainDF[variables].values.T
            y_train = (trainDF[labels].values).reshape(1,-1)
            y_train[y_train>-1.5] = 1
            y_train[y_train<-1.5] = 0

            X_test = testDF[variables].values.T
            y_test = (testDF[labels].values).reshape(1,-1)
            y_test[y_test>-1.5] = 1
            y_test[y_test<-1.5] = 0
            
        
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


angles     = [0,10,20,30,40,50,60,70,80,90]
resolution = ['Coarsest','Coarse']
variables  = ['CfMean','TKE','U','gradP','rmsCp','peakminCp','peakMaxCp','theta','LV0']
labels     = 'meanCp'

angles     = {'LF': [0,10,20,30,40,50,60,70,80,90], 'HF': [0,30,60,90]}
resolution = {'LF': 'Coarsest', 'HF': 'Coarse'}
variables  = ['CfMean','TKE','U','gradP','rmsCp','peakminCp','peakMaxCp','theta','LV0']
labels     = 'meanCp'

datasplit = preprocess_features(angles, resolution, variables, labels, 'MultiFidelity')

X_train, X_test, y_train, y_test = datasplit.split_dataset()

LR = logistic_regression(X_train, X_test, y_train, y_test)
