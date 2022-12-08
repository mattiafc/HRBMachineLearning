import numpy   as np
import pandas  as pd
import tensorflow as tf
import MTLNeuralNetGATTI as gatti
import HRBProbes
import re
import itertools
from datetime import datetime
from sklearn.model_selection import train_test_split
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scipy          import optimize
from scipy.optimize import minimize
from scipy.optimize import Bounds

from joblib         import Parallel, delayed

import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt

font = {'size'   : 15}

matplotlib.rc('font', **font)

class neural_networks:

    X_mean = 0
    X_std  = 1
    
    y_mean = 0
    y_std  = 1    
    
    train_size = 0.80
    dev_size = 0.20
    
    ####################################
    ###   Normalize and split data   ###
    ####################################
    
    def __init__(self, X_train_dev_nn, X_test_nn, y_train_dev, y_test, variables, label, learn_delta = True):
        
        #if dataset == ' Standard'
        
        self.areaIdx  = variables.index('Area')
        self.labelsIdx = [variables.index(l) for l in label]
        self.lvIdx    = variables.index('LV0')
        
        self.nFeatures, self.nSamples = X_train_dev_nn.shape
        
        self.X_mean = np.mean(X_train_dev_nn, axis = 1, keepdims = True)
        self.X_std  = np.std(X_train_dev_nn,  axis = 1, keepdims = True, ddof = 1)+1e-10
        
        #self.y_mean = np.mean(y_train_dev, axis = 1, keepdims = True)
        #self.y_std  = np.std(y_train_dev,  axis = 1, keepdims = True, ddof = 1)+1e-10
        
        self.scaling_factors = [self.X_mean, self.X_std, self.y_mean, self.y_std]
        
        self.X_mean[self.areaIdx] = 0.0
        self.X_std[self.areaIdx]  = 1.0
        
        self.X_train_dev = (X_train_dev_nn - self.X_mean)/self.X_std
        self.y_train_dev = (y_train_dev - self.y_mean)/self.y_std
        
        self.X_test = (X_test_nn  - self.X_mean)/self.X_std
        self.y_test = (y_test - self.y_mean)/self.y_std
        
        
        #self.X_train, self.X_dev, self.y_train, self.y_dev = self.train_dev_split(self.X_train_dev.T, self.y_train_dev.T)
        
        X_train_LF, X_dev_LF, y_train_LF, y_dev_LF = self.train_dev_split(self.X_train_dev[:,:self.nSamples//2].T, self.y_train_dev[:,:self.nSamples//2].T)
        X_train_HF, X_dev_HF, y_train_HF, y_dev_HF = self.train_dev_split(self.X_train_dev[:,self.nSamples//2:].T, self.y_train_dev[:,self.nSamples//2:].T)
        
        self.X_train = np.hstack((X_train_LF, X_train_HF))
        self.y_train = np.hstack((y_train_LF, y_train_HF))
        self.X_dev = np.hstack((X_dev_LF, X_dev_HF))
        self.y_dev = np.hstack((y_dev_LF, y_dev_HF))
        
        #X_train_LF, X_dev_LF, y_train_LF, y_dev_LF = self.train_dev_split(self.X_train_dev[:,:self.nSamples//2].T, self.y_train_dev[:,:self.nSamples//2].T)
        
        #self.X_train = X_train_LF
        #self.y_train = y_train_LF
        #self.X_dev = X_dev_LF
        #self.y_dev = y_dev_LF
        
        print('=====================================================')
        print('Train set X shape: ' + str(self.X_train.shape))
        print('Train set y shape: ' + str(self.y_train.shape))
        print('Dev   set x shape: ' + str(self.X_dev.shape))
        print('Dev   set y shape: ' + str(self.y_dev.shape))
        print('X: ' + str(type(self.X_train)) + ', y: ' + str(type(self.y_train)))
        #print('Layer dimensions: ' + str(layers))
        print('=====================================================\n')
        return
    
    ####################################
    ###  Setup for tf and fit model  ###
    ####################################
        
    def fit_neural_network(self, layers, learning_rate = 0.001, num_epochs = 501, minibatch_size = 128):
        
        self.layers = layers
        
        ### CONSTANTS DEFINING THE MODEL ####
        
        print('=====================================================')
        print('Layer dimensions: ' + str(self.layers))
        print('Learning rate:    ' + str(learning_rate))
        print('Number of epochs: ' + str(num_epochs))
        print('Minibatch size:   ' + str(minibatch_size))
        print('=====================================================\n')
        
        
        train_x = tf.data.Dataset.from_tensor_slices(self.X_train.T)
        train_y = tf.data.Dataset.from_tensor_slices(self.y_train.T)

        dev_x = tf.data.Dataset.from_tensor_slices(self.X_dev.T)
        dev_y = tf.data.Dataset.from_tensor_slices(self.y_dev.T)

        test_x = tf.data.Dataset.from_tensor_slices(self.X_test.T)
        test_y = tf.data.Dataset.from_tensor_slices(self.y_test.T)

        self.parameters, train_cost, dev_cost, costs_plot = gatti.model(train_x, train_y, dev_x, dev_y, test_x, test_y, 
                                        self.layers, self.areaIdx, learning_rate, num_epochs, minibatch_size, self.scaling_factors, self.labelsIdx)
        
        return self.parameters, train_cost, dev_cost, costs_plot
    
    ##########################################################
    ############# Utilities read the saved model #############
    ##########################################################

    def read_model(self, directory = '../MachineLearningOutput/ModelParameters/'):
    
        with open(directory + "setup",'r') as infile:
            for line in infile:
                if 'Layer' in line:
                    layers = eval(line[14:])
                if 'Learning' in line:
                    learning_rate = [float(x) for x in re.findall('[0-9].+', line)][0]
                if 'Number' in line:
                    epochs = list(map(int,re.findall('[0-9]+', line)))[0]
                if 'Minibatch' in line:
                    minibatch = [int(x) for x in re.findall('[0-9]+', line)][0]

        nLayers = sum([len(l) for l in layers])-len(layers)

        self.parameters = {}

        for l in range(1, nLayers+1):
            
            self.parameters['W' + str(l)] = tf.Variable(np.matrix(np.loadtxt(directory + "W"+str(l)+".csv", delimiter = ',')))
            self.parameters['b' + str(l)] = tf.Variable(np.matrix(np.loadtxt(directory + "b"+str(l)+".csv", delimiter = ',')).T)

        return layers, learning_rate, epochs, minibatch
    
    ##########################################
    ###  Make predictions + evaluate RMSE  ###
    ##########################################
    
    def predictions_RMSE(self, seed, readParams = False, test = False):
        self.learn_delta = True
        labels     = ['meanCp','rmsCp','peakMaxCp','peakminCp']
        # First: Compute the RMSE with respect to the various datasets
        
        if readParams == True:
            self.layers, learning_rate, epochs, minibatch = self.read_model(str('../MachineLearningOutput/ModelParameters/' + str(seed) + '/'))
        
        if test == True:
            sets = ['Train', 'Dev', 'Test']
        else:
            sets = ['Train', 'Dev']
        
        for s in sets:
            if s == 'Train':
                given_X     = tf.data.Dataset.from_tensor_slices(self.X_train.T)
                given_Cp_HF = tf.data.Dataset.from_tensor_slices(self.y_train.T)
                _, nGiven = self.X_train.shape
            elif s == 'Dev':
                given_X     = tf.data.Dataset.from_tensor_slices(self.X_dev.T)
                given_Cp_HF = tf.data.Dataset.from_tensor_slices(self.y_dev.T)
                _, nGiven = self.X_dev.shape
            elif s == 'Test':
                given_X     = tf.data.Dataset.from_tensor_slices(self.X_test.T)
                given_Cp_HF = tf.data.Dataset.from_tensor_slices(self.y_test.T)
                _, nGiven = self.X_test.shape
            
            given_dataset = tf.data.Dataset.zip((given_X, given_Cp_HF))
            given_bacth   = given_dataset.batch(nGiven).prefetch(8)
            
            for (X, Y) in given_bacth:
                
            #Need to add the LF data and then calculate direct RMSE    
                if self.learn_delta == True:
                    #Here X is raw Cp [LF; HF] and Y is deltaCp [deltaCp ; 0]
                    # Compute MF error (de-normalize data + RMSE computation)
 
                    _, given_delta_Cp_NN_LH = gatti.forward_propagation(tf.transpose(X), self.parameters, self.layers)
                    if not(s == 'Test'):

                        # Extract LV0, rescale to -+inf, extract LF and HF indices
                        LV_column = tf.gather(X, self.lvIdx, axis=1)
                        rescaled_LV_column = tf.add(tf.multiply(LV_column,self.X_std[self.lvIdx]),self.X_mean[self.lvIdx])
                        abs_dist = np.abs(rescaled_LV_column-0.34)
                        LF_indices = np.where(abs_dist == abs_dist.min())[0]
                        HF_indices = np.where(abs_dist == abs_dist.max())[0]
                    
                        #Gather LF features and HF labels
                        LF_X              = tf.gather(X, LF_indices, axis=0)
                        HF_deltas         = tf.gather(Y, LF_indices, axis=0)
                        given_delta_Cp_NN = tf.gather(given_delta_Cp_NN_LH, LF_indices, axis=1)
                    
                    else:
                        LF_X              = X
                        HF_deltas         = Y
                        given_delta_Cp_NN = given_delta_Cp_NN_LH
                    
                    given_NN_RMSE, given_LF_RMSE  = gatti.compute_cost(tf.transpose(LF_X), tf.transpose(HF_deltas), given_delta_Cp_NN, tf.gather(LF_X, self.areaIdx, axis=1), self.scaling_factors, self.labelsIdx)
                                                                    #  Features:Cp      Labels:DeltaCp   Prediction:MF Cp
            

                else:                
                
                    # Compute MF error (de-normalize data + RMSE computation)
                    _, given_Cp_NN = gatti.forward_propagation(tf.transpose(X), self.parameters, self.layers)
                    given_NN_RMSE, given_LF_RMSE  = gatti.compute_cost(tf.transpose(X), tf.transpose(Y), given_Cp_NN, tf.gather(X, self.areaIdx, axis=1), self.scaling_factors, self.labelsIdx)
                    
            if s == 'Train':
                train_NN_RMSE = given_NN_RMSE
                train_LF_RMSE = given_LF_RMSE
                train_RMSE = [train_NN_RMSE, train_LF_RMSE]
            elif s == 'Dev':
                dev_NN_RMSE = given_NN_RMSE
                dev_LF_RMSE = given_LF_RMSE
                dev_RMSE = [dev_NN_RMSE, dev_LF_RMSE]
            elif s == 'Test':
                test_NN_RMSE = given_NN_RMSE
                test_LF_RMSE = given_LF_RMSE
                test_RMSE = [test_NN_RMSE, test_LF_RMSE]
        
        
        # Second: Compute the ensebmle RMSE
    
        if test == True:
            print('In Predict RMSE')
            print(self.y_test.T)
            X_pred = np.vstack((self.X_train_dev.T, self.X_test.T)).T
            HF_Cp  = np.vstack((self.y_train_dev.T, self.y_test.T)).T
        else:
            X_pred = self.X_train_dev
            HF_Cp  = self.y_train_dev
        
        nFeatures, nPoints = X_pred.shape
        
        pred_X = tf.data.Dataset.from_tensor_slices(X_pred.T)
        Cp_HF  = tf.data.Dataset.from_tensor_slices(HF_Cp.T)
    
        pred_dataset = tf.data.Dataset.zip((pred_X, Cp_HF))
        pred_bacth = pred_dataset.batch(nPoints).prefetch(8)
        
        for (X, Y) in pred_bacth:
            #rescaled_LF_pred = tf.add(tf.multiply(X,self.scaling_factors[1][self.labelsIdx]),self.scaling_factors[0][self.labelsIdx]) 
            #Need to add the LF data and then calculate direct RMSE    
            if self.learn_delta == True:
                    #Here X is raw Cp [LF; HF] and Y is deltaCp [deltaCp ; 0]
                    # Compute MF error (de-normalize data + RMSE computation)
 
                    _, given_delta_Cp_NN_LH = gatti.forward_propagation(tf.transpose(X), self.parameters, self.layers)

                    # Extract LV0, rescale to -+inf, extract LF and HF indices
                    LV_column = tf.gather(X, self.lvIdx, axis=1)
                    rescaled_LV_column = tf.add(tf.multiply(LV_column,self.X_std[self.lvIdx]),self.X_mean[self.lvIdx])
                    abs_dist = np.abs(rescaled_LV_column-0.34)
                    LF_indices = np.where(abs_dist == abs_dist.min())[0]
                    HF_indices = np.where(abs_dist == abs_dist.max())[0]
                    
                    #Gather LF features and HF labels
                    LF_X              = tf.gather(X, LF_indices, axis=0)
                    HF_deltas         = tf.gather(Y, LF_indices, axis=0)
                    given_delta_Cp_NN = tf.gather(given_delta_Cp_NN_LH, LF_indices, axis=1)
                    
                    NN_RMSE, LF_RMSE  = gatti.compute_cost(tf.transpose(LF_X), tf.transpose(HF_deltas), given_delta_Cp_NN, tf.gather(LF_X, self.areaIdx, axis=1), self.scaling_factors, self.labelsIdx)
                    
                    # Rescale LF_X to original space and add delta_Cp_NN to make it Cp_NN
                    scaled_down_Cp_LF = tf.gather(tf.transpose(LF_X), self.labelsIdx, axis=0)
                    Cp_LF    = tf.add(tf.multiply(scaled_down_Cp_LF,self.scaling_factors[1][self.labelsIdx]),self.scaling_factors[0][self.labelsIdx])
                    rescaled_NN_deltas = tf.add(tf.multiply(given_delta_Cp_NN,self.scaling_factors[3]),self.scaling_factors[2])
                    Cp_NN   = tf.add(rescaled_NN_deltas, Cp_LF)
                    
                    # Rescale HF_deltas and add the results to 
                    HF_X  = tf.gather(X, HF_indices, axis=0)
                    scaled_down_HF_Cp = tf.gather(tf.transpose(HF_X), self.labelsIdx, axis=0)
                    Cp_HF = tf.add(tf.multiply(scaled_down_HF_Cp,self.scaling_factors[1][self.labelsIdx]),self.scaling_factors[0][self.labelsIdx])
                    

            else:
                    # Compute MF error (de-normalize data + RMSE computation)
                    _, Cp_NN = gatti.forward_propagation(tf.transpose(X), self.parameters, self.layers)
                    NN_RMSE, LF_RMSE  = gatti.compute_cost(tf.transpose(X), tf.transpose(Y), Cp_NN, tf.gather(X, self.areaIdx, axis=1), self.scaling_factors, self.labelsIdx)
           
        print('=====================================================')
        print('RMSE comparison MF Neural Net vs LF LES over test_dev')
        print('RMSE consider the test set? ' + str(test))
        print('Train set MF RMSE :' +str(train_NN_RMSE))
        print('Train set LF RMSE :' +str(train_LF_RMSE))
        print('Dev   set MF RMSE :' +str(dev_NN_RMSE))
        print('Dev   set LF RMSE :' +str(dev_LF_RMSE))
        if test == True:
            print('Test  set MF RMSE :' +str(test_NN_RMSE))
            print('Test  set LF RMSE :' +str(test_LF_RMSE))
        print('The cumulative results for the sets above are:')   
        print("MF NN      RMSE :" +str(NN_RMSE))
        print("LF LES     RMSE :" +str(LF_RMSE))
        print('=====================================================')
        
        if test == True:
            return X_pred*self.X_std+self.X_mean, Cp_LF.numpy(), Cp_NN.numpy(), Cp_HF.numpy(), NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE, test_RMSE
        else:
            return X_pred*self.X_std+self.X_mean, Cp_LF.numpy(), Cp_NN.numpy(), Cp_HF.numpy(), NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE


    
    ##########################################
    ###  function to make the data split   ###
    ##########################################
    
    def train_dev_split(self, X, y):
        
        if abs(self.train_size + self.dev_size - 1.0) > 1e-6:
            raise Exception("Summation of dataset splits should be 1")
        
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=self.dev_size, random_state=42)
        
        return X_train.T, X_dev.T, y_train.T, y_dev.T
    
    
    
##########################################################
######### Dataset split into train_dev and test; #########
##########################################################
        
        
        
class preprocess_features:

    def __init__(self, angles, resolutions, features, label, dataset = 'Standard'):

        self.angles      = angles
        self.resolutions = resolutions
        self.features    = features
        self.label       = label
        self.dataset     = dataset
        self.test_size    = 0.15

    def split_dataset(self):
        
        if self.dataset == 'Standard':
            
            DF = self.read_file(self.resolutions, self.angles)

            X = DF[variables].values.T
            y = (DF[labels].values).reshape(len(self.label),-1)
            
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
            y_train = (trainLabels[labels].values).T
            
            #print(y_train)
            X_test = testDF[variables].values.T
            y_test = (testLabels[labels].values).T


        elif self.dataset == 'MF-deltaCp0':
            
            """
            Train set: 
                - Use raw LF and HF data at HF angles and predict delta at HF angles (labels)
                - data from LF model @ HF angles
                - label (delta) from HF and LF model @ HF angles
            """
            trainDF1     = self.read_file([self.resolutions['LF']], self.angles['HF']) #Get LF data on HF angles
            trainDF2     = self.read_file([self.resolutions['HF']], self.angles['HF']) #Get HF data on HF angles
            trainDF      = trainDF1.merge(trainDF2, how='outer') #concatenate LF and HF


            trainLabels = trainDF2.subtract(trainDF1) # Get deltaCp
            trainLabels = trainLabels.merge(trainDF2*0, how='outer') # concatenate with 0s because DeltaCp for HF is 0
            
            """
            Test set: 
                - data from LF model @ withheld (test) angles
                - label from HF model @ withheld (test) angles
            """
            try:
                testAngles = list(np.sort(list(set(self.angles['LF']) - set(self.angles['HF']))))
                testDF1     = self.read_file([self.resolutions['LF']], testAngles)
                testDF2     = self.read_file([self.resolutions['HF']], testAngles)
                testDF      = testDF1*1.0

                testLabels = testDF2.subtract(testDF1) 
                X_test = testDF[variables].values.T
                y_test = (testLabels[labels].values).T
            
            except:
                X_test = 0
                y_test = 0
                
            X_train = trainDF[variables].values.T #This is raw Cp [LF ; HF]
            y_train = (trainLabels[labels].values).T #this is the deltaCp
            
            #print(y_train)


        elif self.dataset == 'Standard-deltaCp0':
            
            """
            Train set: 
                - Use raw LF and HF data at HF angles and predict delta at HF angles (labels)
                - data from LF model @ HF angles
                - label (delta) from HF and LF model @ HF angles
            """
            
            DF_LF = self.read_file([self.resolutions['LF']], self.angles) #Get LF data on HF angles
            DF_HF = self.read_file([self.resolutions['HF']], self.angles) #Get HF data on HF angles
            
            labels_LF = DF_HF.subtract(DF_LF) # Get deltaCp
            labels_HF = DF_HF*0 # Get deltaCp            

            X_LF = DF_LF[variables].values.T
            y_LF = labels_LF[labels].values.T
            
            X_HF = DF_HF[variables].values.T
            y_HF = labels_HF[labels].values.T
            
            X_train_LF, X_test_LF, y_train_LF, y_test_LF = self.ordinary_train_test(X_LF.T, y_LF.T)
            X_train_HF, X_test_HF, y_train_HF, y_test_HF = self.ordinary_train_test(X_HF.T, y_HF.T)
        
            X_train = np.hstack((X_train_LF, X_train_HF))
            y_train = np.hstack((y_train_LF, y_train_HF))
            X_test = np.hstack((X_test_LF, X_test_HF))
            y_test = np.hstack((y_test_LF, y_test_HF))
            
        #print('=====================================================')
        #print('Total number of samples is: ' + str(X_train.shape[1]+X_test.shape[1]))
        #print('Training set is ' + str(X_train.shape[1]) + ' samples')
        #print('Test     set is ' +str(X_test.shape[1])  + ' samples')
        #print('=====================================================\n')

        return X_train, X_test, y_train, y_test

    def ordinary_train_test(self, X, y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=4)
        
        return X_train.T, X_test.T, y_train.T, y_test.T
    
    
    def read_file(self, resolutions, angles):
            
        data = []
        for res in resolutions:
        
            for ang in angles:
                
                data.append(pd.read_csv(str('Features/' + res + str(ang))))
        
        return pd.concat(data, axis=0) 
    
    
    
##########################################################
###### Grid search for hyperparameter optimization #######
##########################################################   



def parallelGridSearch(seed, X_train_dev, X_test, y_train_dev, y_test, variables, labels, test = False):

    np.random.seed(seed)
    
    n_shared_layers = np.random.randint(0, high = 5)
    shared_layers   = (np.random.randint(15, size = int(n_shared_layers))+2).tolist()
    shared_layers.insert(0, X_train_dev.shape[0])
    
    #Compatibility between shared and dedicated
    shared_last_layer = (np.random.randint(15)+2)
    shared_layers.append(shared_last_layer)
    
    n_dedicated_layers = np.random.randint(0, high = 4)
    dedicated_layers   = (np.random.randint(15, size = int(n_dedicated_layers))+2).tolist()
    
    dedicated_layers.insert(0, shared_last_layer)
    dedicated_layers.append(1)
    
    #layers_list = [[11, 15, 15, 15, 15, 14],[14, 10, 10, 10, 1],[14, 10, 8, 11, 1],[14, 10, 8, 11, 1],[14, 10, 8, 11, 1]]
    layers_list = [shared_layers, dedicated_layers, dedicated_layers, dedicated_layers, dedicated_layers]
    
    learning_rate = 10**np.random.uniform(-5.0,-2.5)
    n_epochs      = 701
    batch_size    = int(2**np.round(np.random.uniform(6.0, 8.1)))

    neuralNet = neural_networks(X_train_dev, X_test, y_train_dev, y_test, variables, labels)

    parameters, train_cost, dev_cost, costs_plot = neuralNet.fit_neural_network(layers_list, learning_rate, n_epochs, batch_size)
    
    #plt.savefig('../MachineLearningOutput/Plots/WeightAndGrad/Label:%s,Seed:%d.png' %(labels, seed))
    #plt.close()
    
    directory = '../MachineLearningOutput/ModelParameters/' + str(seed) + '/'
    
    try:
        os.mkdir(directory)
    except:
        try:
            os.mkdir(directory[:-1] + '(1)')
        except:
            with open('../MachineLearningOutput/GridsearchThemis.dat', 'a+') as out:
                out.write('ITERATION ' + str(seed) + ' OVERWROTE THE CONTENT OF ModelParameters/' + str(seed) + '\n')
        
    
    for key, value in parameters.items():
        if 'b' in key or 'W' in key:
            np.savetxt(directory + key + '.csv', value.numpy(), delimiter = ',')
            
    with open(directory +'/setup', 'w+') as out:
        out.write('Layers setup: '    +str(layers_list) + '\n')
        out.write('Learning rate: '   +str(learning_rate) + '\n')
        out.write('Number of epochs: '+str(n_epochs) + '\n')
        out.write('Minibatch size: '  +str(batch_size) + '\n')
    
    if test == False:
        X_pred, Cp_LF, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE  = neuralNet.predictions_RMSE(0, readParams = False, test=test)
    
    elif test == True:
        X_pred, Cp_LF, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE, test_RMSE  = neuralNet.predictions_RMSE(0,readParams = False, test=test)


    with open('../MachineLearningOutput/GridsearchThemis.dat', 'a+') as out:
        out.write('========================================\n')
        out.write('Seed number         : %d\n' %seed)
        out.write('Layer dimensions    : ' + str(layers_list)+'\n')
        out.write('Learning rate       : %f\n' %learning_rate)
        out.write('Number of epochs    : %d\n' %n_epochs)
        out.write('Minibatch size      : %d\n' %batch_size)
        out.write("MF NN  integral RMSE: " + str(NN_RMSE.numpy()) + '\n')
        out.write("LF LES integral RMSE: " + str(LF_RMSE.numpy()) + '\n')
        out.write("Train NN/LF     RMSE: " + str(train_RMSE[0].numpy()) + ', ' +  str(train_RMSE[1].numpy()) + '\n')
        out.write("Dev   NN/LF     RMSE: " + str(dev_RMSE[0].numpy()) + ', ' +  str(dev_RMSE[1].numpy()) + '\n')
        if test == True:
            out.write("Test  NN/LF     RMSE:" + str(test_RMSE[0].numpy()) + ', ' +  str(test_RMSE[1].numpy())+ '\n')
        out.write('========================================\n')
    
    #costs_plot = np.asarray(costs_plot)
    #plt.plot(costs_plot[:,2], costs_plot[:,0], label = 'Train')
    #plt.plot(costs_plot[:,2], costs_plot[:,1], label = 'Dev')
    #plt.xlabel('# of Epochs')
    #plt.ylabel('Cost')
    #plt.title('Label: %s, Seed: %d' %(labels, seed))
    #plt.legend(frameon=False)
    #plt.savefig('../MachineLearningOutput/Plots/Costs/Label:%s,Seed:%d.png' %(labels, seed))
    #plt.close()
    #plt.plot(np.squeeze(lastStep[-2:-1]-t0)*tStep*np.ones(y.shape),y,color='lime',linewidth = 2.0)
    
    return
    
    
##########################################################
######## Utilities to save and read dat for plots ########
##########################################################



def saveToDat(patches, angles, resolution, variables, labels, trainDF, X_pred, Cp_NN, Cp_HF, Cp_LF):

    predDF = pd.DataFrame(trainDF, columns = variables)
    
    predDF[[l + 'LF' for l in labels]] = Cp_LF.T
    predDF[[l + 'NN' for l in labels]] = Cp_NN.T
    predDF[[l + 'HF' for l in labels]] = Cp_HF.T
    cumulativeDF = pd.merge(trainDF,predDF)

    LFEntries = ['# x', 'y', 'z'] + [l + 'LF' for l in labels] + ['Area']
    NNEntries = ['# x', 'y', 'z'] + [l + 'NN' for l in labels] + ['Area']
    HFEntries = ['# x', 'y', 'z'] + [l + 'HF' for l in labels] + ['Area']

    for ang in angles:
        
        angleData = cumulativeDF.loc[cumulativeDF['theta'] == ang]
        
        for pl in patches:
        
            if   pl == 'L':
                df = angleData.loc[abs(angleData['z'] - 0.15) < 1e-6]
                
            elif pl == 'W':
                df = angleData.loc[abs(angleData['z'] + 0.15) < 1e-6]
                
            elif pl == 'T':
                df = angleData.loc[abs(angleData['y'] - 2.0) < 1e-6]
                
            elif pl == 'F':
                df = angleData.loc[abs(angleData['# x']) < 1e-6]
                
            elif pl == 'R':
                df = angleData.loc[abs(angleData['# x'] - 1.0) < 1e-6]
                
            else:
                raise Exception("PL is sughellamento totale")
            
            LFPred = df[LFEntries].values
            NNPred = df[NNEntries].values
            HFPred = df[HFEntries].values
            
            fNameLF = str('../MachineLearningOutput/LFPred/' + resolution['LF'] + str(ang) + pl + '.dat')
            fNameNN = str('../MachineLearningOutput/NNPred/NeuralNet' + str(ang) + pl + '.dat')
            fNameHF = str('../MachineLearningOutput/HFPred/' + resolution['HF'] + str(ang) + pl + '.dat')
            
            np.savetxt(fNameLF, LFPred, header = str(LFEntries))
            np.savetxt(fNameNN, NNPred, header = str(NNEntries))
            np.savetxt(fNameHF, HFPred, header = str(HFEntries))


def readDat(patches, angles, deltas, labels, directory):

    roundoff = 12
    probes = {}

    for lvl in deltas:
        for ang in angles:
            for pl in patches:
                    
                dictKey = lvl + str(ang) + pl
                
                #fName = str('probesToDat/'+dictKey+'.dat')
                fName = str(directory+dictKey+'.dat')
                
                temp = np.loadtxt(fName, skiprows=1)
                
                coords = np.transpose(np.array([temp[:,0],temp[:,1],temp[:,2]]))
                
                probes[dictKey] = {'coords': coords}
                
                cont = 3
                
                for l in labels:
                    
                    probes[dictKey][l] = temp[:,cont]
                    cont += 1
                probes[dictKey]['Area'] = temp[:,cont]

    return probes


#def optimizer_wrap(params, *args):
    
    #X_train_dev, X_test, y_train_dev, y_test, variables, labels = args
    
    #learning_rate = 10**params[0]
    #batch_size    = 2**round(params[1])
    #layer_size    = [int(params[2])]
    #n_layers      = int(params[3])
    #n_epochs      = 701
    #layers        = layer_size*n_layers
    
    #layers.insert(0, X_train_dev.shape[0])
    #layers.append(1)
    
    #print(X_train_dev)
    #print(X_test)
    #print(variables)
    #print(labels)
    
    #print(learning_rate)
    #print(batch_size)
    #print(layer_size)
    #print(layers)
    
    #neuralNet = neural_networks(X_train_dev, X_test, y_train_dev, y_test, variables, labels)
    
    #parameters, train_cost, dev_cost, costs_plot = neuralNet.fit_neural_network(layers, learning_rate, n_epochs, batch_size)
    
    #plt.close()

    #X_pred, Cp_LF, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE  = neuralNet.predictions_RMSE(False, False)
    
    #print('========================================')
    #print('Layer dimensions    : ' + str(layers)+'\n')
    #print('Learning rate       : %f' %learning_rate)
    #print('Number of epochs    : %d' %n_epochs)
    #print('Minibatch size      : %d' %batch_size)
    #print("MF NN  integral RMSE: %f" %NN_RMSE)
    #print("LF LES integral RMSE: %f" %LF_RMSE)
    #print("Train/dev       RMSE: %f, %f" %(train_cost, dev_cost))
    #print('========================================\n')

    #return dev_cost
    
def computePdf(patches, angles, resolution, quantities, pdfMetric, scale, directory):
    
    my_dpi = 100
                
    plt.figure(figsize=(3360/my_dpi, 630/my_dpi), dpi=my_dpi)

    for qty in quantities:
        
        metric = []
        area   = []
            
        if qty == 'meanCp':
            stringa = 'Mean Cp'
            delta = 0.1
            plt.subplot(141)
            
        elif qty == 'rmsCp':
            stringa = 'Rms Cp'
            delta = 0.05
            plt.subplot(142)
            
        elif qty == 'peakMaxCp':
            stringa = 'Peak Max Cp'
            delta = 0.2
            plt.subplot(143)
            
        elif qty == 'peakminCp':
            stringa = 'Peak Min Cp'
            delta = 0.2
            plt.subplot(144)
        
        for pl in patches:
            for ang in angles:
            
                dictKey = resolution + str(int(ang)) + pl
            
                metric.append(pdfMetric[dictKey][qty])
                area.append(pdfMetric[dictKey]['tapsArea'])
            
        pdfToPlot = np.squeeze(list(itertools.chain.from_iterable(metric)))
        AreaToCompute = np.squeeze(list(itertools.chain.from_iterable(area)))
        okPoints = [c for c in pdfToPlot if c>-delta and c<delta]
        idx = []
        for error in okPoints:
            idx.append(np.squeeze(np.where(pdfToPlot == error)[0]))

            
        print('Points with error within +-' + str(delta) + qty + ' is %s%%' %str(round(100*len(idx)/len(pdfToPlot),2)))
        print('Area within +-' + str(delta) + qty + ' is %s%%' %str(round(100*sum(AreaToCompute[idx])/sum(AreaToCompute),2)))
        #print(idx[-1:])
        
        occ, x, _ = plt.hist(pdfToPlot, bins = 'auto')
        nElem = 100
        
        xmin = np.linspace(-delta,-delta,nElem)
        ymin = np.linspace(0,max(occ),nElem)
        
        xMax = np.linspace(delta,delta,nElem)
        yMax = np.linspace(0,max(occ),nElem)
        
        plt.xlabel('$\Delta$')
        plt.ylabel('Occurrence')
        plt.plot(xmin,ymin,'k--', linewidth = 2.5)
        plt.plot(xMax,yMax,'k--', linewidth = 2.5)
        plt.xlim([scale[qty][0],scale[qty][1]])
        plt.xticks([scale[qty][0], -delta, delta, scale[qty][1]])
        plt.title(stringa)
        
    plt.savefig(directory, bbox_inches='tight')
    #plt.show(block=True)
    plt.close('all')
            
    return


##########################################################
########                 CODE BODY                ########
##########################################################

#angles     = {'LF': [0,10,20,30,40,50,60,70,80,90], 'HF': [10,30,50,70,90]}
#angles     = {'LF': [0,10,20,30,40,50,60,70,80,90], 'HF': [0,10,30,40,60,70,90]}

angles     = {'LF': [0,10,20,30,40,50,60,70,80,90], 'HF': [0,10,20,30,40,50,60,70,80,90]}
resolution = {'LF': 'Coarsest', 'HF': 'Coarse'}
variables  = ['LV0','CfMean','TKE','U','gradP','UDotN','theta','meanCp','rmsCp','peakminCp','peakMaxCp','Area']
#labels     = ['meanCp']
labels     = ['meanCp','rmsCp','peakMaxCp','peakminCp']

patches = {'F':'front','L':'leeward','R':'rear','T':'top','W':'windward'}

CpScale = {'0' :{'meanCp': [-1.2, 1.0],'rmsCp': [0, 0.36],'peakMaxCp': [-0.5, 1.7],'peakminCp': [-2.5, 0.5], 'meanCpRMSE': [0, 0.01],'rmsCpRMSE': [0, 0.003],'peakMaxCpRMSE': [0, 0.01],'peakminCpRMSE': [0, 0.025]},
           '10':{'meanCp': [-1.2, 1.0],'rmsCp': [0, 0.36],'peakMaxCp': [-0.5, 1.7],'peakminCp': [-2.5, 0.5], 'meanCpRMSE': [0, 0.01],'rmsCpRMSE': [0, 0.003],'peakMaxCpRMSE': [0, 0.01],'peakminCpRMSE': [0, 0.025]},
           '20':{'meanCp': [-1.2, 1.0],'rmsCp': [0, 0.36],'peakMaxCp': [-0.5 ,1.7],'peakminCp': [-2.5, 0.5], 'meanCpRMSE': [0, 0.01],'rmsCpRMSE': [0, 0.003],'peakMaxCpRMSE': [0 ,0.01],'peakminCpRMSE': [0, 0.025]},
           '30':{'meanCp': [-1.5, 1.0],'rmsCp': [0, 0.45],'peakMaxCp': [-0.5, 1.7],'peakminCp': [-3.0, 0.5], 'meanCpRMSE': [0, 0.01],'rmsCpRMSE': [0, 0.003],'peakMaxCpRMSE': [0, 0.01],'peakminCpRMSE': [0, 0.025]},
           '40':{'meanCp': [-2.0, 1.0],'rmsCp': [0, 0.55],'peakMaxCp': [-1.0, 1.5],'peakminCp': [-4.0, 0.3], 'meanCpRMSE': [0, 0.01],'rmsCpRMSE': [0, 0.003],'peakMaxCpRMSE': [0, 0.01],'peakminCpRMSE': [0, 0.025]},
           '50':{'meanCp': [-2.0, 1.0],'rmsCp': [0, 0.55],'peakMaxCp': [-1.0, 1.7],'peakminCp': [-4.0, 0.3], 'meanCpRMSE': [0, 0.01],'rmsCpRMSE': [0, 0.003],'peakMaxCpRMSE': [0, 0.01],'peakminCpRMSE': [0, 0.025]},
           '60':{'meanCp': [-1.2, 1.0],'rmsCp': [0, 0.45],'peakMaxCp': [-0.5, 1.5],'peakminCp': [-3.5, 0.3], 'meanCpRMSE': [0, 0.01],'rmsCpRMSE': [0, 0.003],'peakMaxCpRMSE': [0, 0.01],'peakminCpRMSE': [0, 0.025]},
           '70':{'meanCp': [-1.2, 1.0],'rmsCp': [0, 0.45],'peakMaxCp': [-0.5, 1.5],'peakminCp': [-3.0, 0.5], 'meanCpRMSE': [0, 0.01],'rmsCpRMSE': [0, 0.003],'peakMaxCpRMSE': [0, 0.01],'peakminCpRMSE': [0, 0.025]},
           '80':{'meanCp': [-1.0, 1.0],'rmsCp': [0, 0.35],'peakMaxCp': [-0.3, 1.5],'peakminCp': [-2.5, 0.5], 'meanCpRMSE': [0, 0.01],'rmsCpRMSE': [0, 0.003],'peakMaxCpRMSE': [0, 0.01],'peakminCpRMSE': [0, 0.025]},
           '90':{'meanCp': [-1.0, 1.0],'rmsCp': [0, 0.30],'peakMaxCp': [-0.3 ,1.5],'peakminCp': [-2.0, 0.5], 'meanCpRMSE': [0, 0.01],'rmsCpRMSE': [0, 0.003],'peakMaxCpRMSE': [0 ,0.01],'peakminCpRMSE': [0, 0.025]}}

pdfScale = {'meanCp': [-0.3, 0.3],'rmsCp': [-0.2, 0.2],'peakMaxCp': [-0.6, 0.6],'peakminCp': [-1.0, 1.0]}

#datasplit = preprocess_features(angles, resolution, variables, labels, 'MF-deltaCp0')
datasplit = preprocess_features(angles, resolution, variables, labels, 'MF-deltaCp0')
#datasplit = preprocess_features(angles, resolution, variables, labels, 'MultiFidelity')

#datasplit = preprocess_features([0,10,20,30,40,50,60,70,80,90], resolution, variables, labels, 'Standard-deltaCp0')

X_train_dev, X_test, y_train_dev, y_test = datasplit.split_dataset()

#print(X_train_dev.shape)
#print(X_test.shape)
#print(y_train_dev.shape)
#print(y_test.shape)

    
#with open('../MachineLearningOutput/GridsearchThemis.dat', 'a+') as out:
   #now = datetime.now()
   #out.write('\n'*10+'Gridsearch performed on ' + str(now.strftime("%d %m %Y, %H:%M:%S"))+ '\n'*10)
    
#_ = Parallel(n_jobs= 8)(delayed(parallelGridSearch)(seed, X_train_dev, X_test, y_train_dev, y_test, variables, labels)
                           #for seed in [37, 116, 96, 246, 59, 241, 78, 163, 69, 189, 145, 10])

#bounds = [(-2,-4),(4.51,7.49),(3.51,16.49), (3.51,8.49)]
#res = optimize.differential_evolution(optimizer_wrap, bounds, args = (X_train_dev, X_test, y_train_dev, y_test, variables, labels),
                                      #popsize = 24, seed = 4, workers = 12, integrality = [False, True, True, True])

neuralNet = neural_networks(X_train_dev, X_test, y_train_dev, y_test, variables, labels)
#parameters = neuralNet.fit_neural_network([[12, 15, 15, 15, 15, 14],[14, 10, 8, 11, 1],[14, 10, 8, 11, 1],[14, 10, 8, 11, 1],[14, 10, 8, 11, 1]], 0.0008193, 701, 64)
#plt.close()

#################################################

test = False
readParams = True

seed = 96

if test == True:

    X_pred, Cp_LF, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE = neuralNet.predictions_RMSE(seed, readParams, False)
    X_pred, Cp_LF, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE, test_RMSE = neuralNet.predictions_RMSE(seed, readParams, test)
print('Seed is '+str(seed))
test = False

X_pred, Cp_LF, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE = neuralNet.predictions_RMSE(seed, readParams, test)

trainDF = datasplit.read_file([resolution['LF']], angles['HF'])

saveToDat(patches, angles['HF'], resolution, variables, labels, trainDF, X_pred, Cp_NN, Cp_HF, Cp_LF)

#Neural nets has been fitted, used to predict, and the output has been saved into a .dat



####################################################
###### HERE YOU CAN PLOT PREDICTIONS AND RMSE ######
####################################################

LFres = [resolution['LF']]
NNres = ['NeuralNet']
HFres = [resolution['HF']]

#for ang in angles['HF']:
    
    #LFresults = readDat(patches, [ang], LFres, labels, directory = '../MachineLearningOutput/LFPred/')
    #HRBProbes.plotQty(LFresults, LFres, [ang], patches, labels, CpScale[str(ang)], directory = '../MachineLearningOutput/Plots/', resCompare = False)

    #NNresults = readDat(patches, [ang], NNres, labels, directory = '../MachineLearningOutput/NNPred/')
    #HRBProbes.plotQty(NNresults, NNres, [ang], patches, labels, CpScale[str(ang)], directory = '../MachineLearningOutput/Plots/', resCompare = False)

    #HFresults = readDat(patches, [ang], HFres, labels, directory = '../MachineLearningOutput/HFPred/')
    #HRBProbes.plotQty(HFresults, HFres, [ang], patches, labels, CpScale[str(ang)], directory = '../MachineLearningOutput/Plots/', resCompare = False)

#for ang in angles['HF']:

    #NNresults = readDat(patches, [ang], NNres, labels, directory = '../MachineLearningOutput/NNPred/')
    #HFresults = readDat(patches, [ang], HFres, labels, directory = '../MachineLearningOutput/HFPred/')
        
    #for l in labels:
    
        #for pl in patches:
                
            #dictKeyNN = NNres[0] + str(int(ang)) + pl
            #dictKeyHF = HFres[0] + str(int(ang)) + pl
        
            #if max(abs(NNresults[dictKeyNN]['Area'] - HFresults[dictKeyHF]['Area'])) > 1e-10:
                #raise Exception('You sughed the area baby!!')
        
            #RMSEcomp = np.sqrt(np.power(NNresults[dictKeyNN][l] - HFresults[dictKeyHF][l],2)*NNresults[dictKeyNN]['Area'])
            #NNresults[dictKeyNN][l+'RMSE'] = RMSEcomp
        
    #HRBProbes.plotQty(NNresults, NNres, [ang], patches, [l+'RMSE' for l in labels], CpScale[str(ang)], directory = '../MachineLearningOutput/Plots/', resCompare = False)


pdfMetric = {}

for ang in angles['HF']:
    for l in labels:
        for pl in patches:
            
            dictKeyNN = NNres[0] + str(int(ang)) + pl
            dictKeyLF = LFres[0] + str(int(ang)) + pl 
            
            pdfMetric[dictKeyNN] = {}     
            pdfMetric[dictKeyLF] = {}     

for ang in angles['HF']:

    LFresults = readDat(patches, [ang], LFres, labels, directory = '../MachineLearningOutput/LFPred/')
    NNresults = readDat(patches, [ang], NNres, labels, directory = '../MachineLearningOutput/NNPred/')
    HFresults = readDat(patches, [ang], HFres, labels, directory = '../MachineLearningOutput/HFPred/')
            
    for l in labels:
        
        for pl in patches:
            
            dictKeyLF = LFres[0] + str(int(ang)) + pl        
            dictKeyNN = NNres[0] + str(int(ang)) + pl
            dictKeyHF = HFres[0] + str(int(ang)) + pl
            
            pdfMetric[dictKeyNN][l]          = np.abs(NNresults[dictKeyNN][l] - HFresults[dictKeyHF][l])*np.sign(np.abs(NNresults[dictKeyNN][l]) - np.abs(HFresults[dictKeyHF][l]))
            pdfMetric[dictKeyNN]['tapsArea'] = NNresults[dictKeyNN]['Area']
            
            pdfMetric[dictKeyLF][l]          = np.abs(LFresults[dictKeyLF][l] - HFresults[dictKeyHF][l])*np.sign(np.abs(LFresults[dictKeyLF][l]) - np.abs(HFresults[dictKeyHF][l]))
            pdfMetric[dictKeyLF]['tapsArea'] = NNresults[dictKeyNN]['Area']

computePdf(patches, angles['HF'], NNres[0], labels, pdfMetric, pdfScale, '../MachineLearningOutput/Plots/PdfError/NN')
computePdf(patches, angles['HF'], LFres[0], labels, pdfMetric, pdfScale, '../MachineLearningOutput/Plots/PdfError/LF')


