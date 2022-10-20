import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

np.random.seed(2)

def initialize_parameters(layers):
    
    """
    Arguments:
    layers -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layers[l], layers[l-1])
                    bl -- bias vector of shape (layers[l], 1)
                  
    """
    
    parameters = {}
    L = len(layers)

    for l in range(1, L):
    
        parameters['W' + str(l)] = tf.Variable(np.random.randn(layers[l], layers[l-1])/np.sqrt(layers[l-1]))
        parameters['b' + str(l)] = tf.Variable(np.zeros((layers[l], 1)))
        
        ## YOUR CODE ENDS HERE
        #print(parameters['W' + str(l)].shape)
        assert(parameters['W' + str(l)].shape == (layers[l], layers[l - 1]))
        assert(parameters['b' + str(l)].shape == (layers[l], 1))

        
    return parameters

def forward_propagation(X, parameters, layers):
    
    parameters['A0'] = tf.constant(X)
    
    L = len(layers)-1

    for l in range(1, L):
        
        parameters['Z' + str(l)] = tf.add(tf.matmul(parameters['W' + str(l)], parameters['A' + str(l-1)]), parameters['b' + str(l)])
        parameters['A' + str(l)] = tf.keras.activations.relu(parameters['Z' + str(l)])
        
        
        assert(parameters['A' + str(l)].shape[0] == parameters['b' + str(l)].shape[0])
        assert(parameters['A' + str(l)].shape[1] == X.shape[1])
        assert(parameters['A' + str(l)].shape == parameters['Z' + str(l)].shape)
    
    parameters['Z' + str(L)] = tf.add(tf.matmul(parameters['W' + str(L)], parameters['A' + str(L-1)]), parameters['b' + str(L)])
    parameters['A' + str(L)] = tf.keras.activations.relu(parameters['Z' + str(L)])
        
        
    assert(parameters['A' + str(L)].shape[0] == parameters['b' + str(L)].shape[0])
    assert(parameters['A' + str(L)].shape[1] == X.shape[1])
    assert(parameters['A' + str(L)].shape == parameters['Z' + str(L)].shape)
    
    
    return parameters


def compute_cost(labels, parameters, layers, area):
    
    L = len(layers)-1
    
    y_hat = parameters['A' + str(L)]
    
    ATimesSq = tf.multiply(tf.pow(tf.subtract(y_hat, labels),2), area)
    
    cost = tf.sqrt( tf.divide(tf.reduce_sum(ATimesSq),tf.reduce_sum(area)) )
    
    return tf.sqrt(cost)

def model(X_train, Y_train, X_test, Y_test, layers, learning_rate = 0.001,
          num_epochs = 300, minibatch_size = 0, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    costs = []                                        # To keep track of the cost
    train_acc = []
    test_acc = []
        
    parameters = initialize_parameters(layers)    
    
    train_vars = []
    
    for l in range(1,len(layers)):
        train_vars.append(parameters['W' + str(l)])
        train_vars.append(parameters['b' + str(l)])

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))
    
    train_size = tf.data.experimental.cardinality(X_train).numpy()
    
    test_size  = tf.data.experimental.cardinality(X_test).numpy()
    
    # We can get the number of elements of a dataset using the cardinality method
    m = dataset.cardinality().numpy()
    
    if minibatch_size == 0:
        minibatch_size = train_size
    
    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(test_size).prefetch(8)
    
    #X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step    
    #Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster
    
    # Do the training loop
    for epoch in range(num_epochs):
        
        epoch_cost = 0.
        
        for (minibatch_X, minibatch_Y) in minibatches:
            
            with tf.GradientTape() as tape:
                
                # Forward propagation
                parameters = forward_propagation(tf.transpose(minibatch_X), parameters, layers)

                # Loss & Accuracy computation
                minibatch_cost = compute_cost(tf.transpose(minibatch_Y), parameters, layers, tf.gather(minibatch_X, layers[0]-1, axis=1))
                
            
            trainable_variables = train_vars
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost
        
        # We divide the epoch cost over the number of samples
        #epoch_cost /= m

        # Print the cost every 10 epochs
        if print_cost == True and epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            #print ("Train accuracy %f" % train_accuracy)
            
            for (t_minibatch_X, t_minibatch_Y) in test_minibatches:
            
                parameters = forward_propagation(tf.transpose(t_minibatch_X), parameters, layers)
                cost_test  = compute_cost(tf.transpose(t_minibatch_Y), parameters, layers, tf.gather(t_minibatch_X, layers[0]-1, axis=1))
                
            print("Test Cost %f" %cost_test)

            costs.append(epoch_cost)

    return parameters, costs, train_acc, test_acc

class preprocess_features:
    
    train_size = 0.60
    test_size = 0.40

    def __init__(self, angles, resolutions, features, label, dataset = 'Standard'):

        self.angles      = angles
        self.resolutions = resolutions
        self.features    = features
        self.label       = label
        self.dataset     = dataset
        self.thresh      = 0.2

    def split_dataset(self):
        
        if self.dataset == 'Standard':
            
            DF = self.read_file(self.resolutions, self.angles)

            X = DF[variables].values.T
            y = (DF[labels].values).reshape(1,-1)
        
            #y[y>self.thresh] = 1
            #y[y<self.thresh] = 0
            
            X_train, X_test, y_train, y_test = self.ordinary_train_test(X.T, y.T)
        
        if self.dataset == 'MultiFidelity':
            
            LFTrainDF = self.read_file([self.resolutions['LF']], self.angles['LF'])
            HFTrainDF = self.read_file([self.resolutions['HF']], self.angles['HF'])
            
            trainDF = pd.concat([LFTrainDF, HFTrainDF], axis=0)
            print(trainDF)
            testAngles = np.sort(list(set(self.angles['LF']) - set(self.angles['HF'])))
            testDF = self.read_file([self.resolutions['HF']], testAngles)
            print(testDF)

            X_train = trainDF[variables].values.T
            y_train = (trainDF[labels].values).reshape(1,-1)
            #y_train[y_train>self.thresh] = 1
            #y_train[y_train<self.thresh] = 0

            X_test = testDF[variables].values.T
            y_test = (testDF[labels].values).reshape(1,-1)
            #y_test[y_test>self.thresh] = 1
            #y_test[y_test<self.thresh] = 0
            
        
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

    
#import numpy as np
#import pandas as pd
#import tensorflow as tf
#import matplotlib.pyplot as plt
#import scipy

#np.random.seed(1)

#angles     = [0,10,20,30,40,50,60,70,80,90]
#resolution = ['Coarsest','Coarse']
#variables  = ['CfMean','TKE','U','gradP','meanCp','peakminCp','peakMaxCp','theta','LV0','Area']
#labels     = 'rmsCp'

#angles     = {'LF': [0,10,20,30,40,50,60,70,80,90], 'HF': [0,30,60,90]}
#resolution = {'LF': 'Coarsest', 'HF': 'Coarse'}
#variables  = ['CfMean','TKE','U','gradP','meanCp','peakminCp','peakMaxCp','theta','LV0','Area']
#labels     = 'rmsCp'

#datasplit = preprocess_features(angles, resolution, variables, labels, 'MultiFidelity')

#X_train, X_test, y_train, y_test = datasplit.split_dataset()

        
#print('=====================================================')
#print('Train set X shape: ' + str(X_train.shape))
#print('Test  set y shape: ' + str(y_train.shape))
#print('Train set x shape: ' + str(X_test.shape))
#print('Test  set y shape: ' + str(y_test.shape))
#print('X: ' + str(type(X_train)) + ', y: ' + str(type(y_train)))
#print('=====================================================\n')

#n_x = X_train.shape[0]

#train_x = tf.data.Dataset.from_tensor_slices(X_train.T)

#test_x = tf.data.Dataset.from_tensor_slices(X_test.T)

#train_y = tf.data.Dataset.from_tensor_slices(y_train.T)

#test_y = tf.data.Dataset.from_tensor_slices(y_test.T)

#### CONSTANTS DEFINING THE MODEL ####
#layers = [n_x,10,7,6,5,4,1]

#model(train_x, train_y, test_x, test_y, layers)



