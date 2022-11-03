import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt

from scipy.signal import convolve2d

font = {'size'   : 15}

matplotlib.rc('font', **font)

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
        parameters['b' + str(l)] = tf.Variable(np.random.randn(layers[l], 1))
        #parameters['b' + str(l)] = tf.Variable(np.zeros((layers[l], 1)))
        
        #assert(parameters['W' + str(l)].shape == (layers[l], layers[l - 1]))
        #assert(parameters['b' + str(l)].shape == (layers[l], 1))

        
    return parameters

def forward_propagation(X, parameters, layers):
    
    A = tf.constant(X)
    
    L = len(layers)-1

    for l in range(1, L):
        
        parameters['Z' + str(l)] = tf.add(tf.matmul(parameters['W' + str(l)], A), parameters['b' + str(l)])
        A = tf.keras.activations.tanh(parameters['Z' + str(l)])
        
        #assert(A.shape[0] == parameters['b' + str(l)].shape[0])
        #assert(A.shape[1] == X.shape[1])
        #assert(A.shape == Z.shape)
    
    parameters['Z' + str(L)] = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])
    #A = tf.keras.activations.tanh(parameters['Z' + str(L)])
    A = parameters['Z' + str(L)]
           
    #assert(A.shape[0] == parameters['b' + str(L)].shape[0])
    #assert(A.shape[1] == X.shape[1])
    #assert(A.shape == Z.shape) 
    
    return parameters, A

def weight_statistics(parameters, layers, code):
    
    L = len(layers)
    mean = [None]*(L-1)
    std  = [None]*(L-1)
    
    for l in range(1, L):
        
        mean[l-1] = np.mean(parameters[code + str(l)].numpy().reshape(-1,1))
        std[l-1]  = np.std(parameters[code + str(l)].numpy().reshape(-1,1), ddof = 1)
        
    return mean, std

def grads_statistics(gradients, layers):
    
    L = len(layers)
    mean = [None]*(L-1)
    std  = [None]*(L-1)
    
    for l in range(0,(L-1)*2,2):
        
        idx = l//2
        
        #Check whether or not we're reading the weight gradient
        assert(gradients[l].numpy().shape == (layers[idx+1], layers[idx]))
        
        mean[idx] = np.mean(gradients[l].numpy().reshape(-1,1))
        std[idx]  = np.std(gradients[l].numpy().reshape(-1,1), ddof = 1)
        
    return mean, std
        

def compute_cost(labels, y_hat, layers, area):
    
    ATimesSq = tf.multiply(tf.pow(tf.subtract(y_hat, labels),2), area)
    
    RMSE = tf.sqrt(tf.divide(tf.reduce_sum(ATimesSq),tf.reduce_sum(area)))
    
    return RMSE

def model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, layers, areaIdx,
          learning_rate, num_epochs, minibatch_size, test_check = False):
    
    # Initalize costs, variables, and optimizer
    
    weight_mean = []
    weight_std  = []
    linear_mean = []
    linear_std  = []
    grad_mean = []
    grad_std  = []
    
    
    parameters = initialize_parameters(layers)
    
    train_vars = []
    costs_plot = []
    
    for l in range(1,len(layers)):
        train_vars.append(parameters['W' + str(l)])
        train_vars.append(parameters['b' + str(l)])

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # Building training and dev set batches
    
    train_dataset = tf.data.Dataset.zip((X_train, Y_train))
    dev_dataset   = tf.data.Dataset.zip((X_dev,   Y_dev))
    test_dataset  = tf.data.Dataset.zip((X_test,  Y_test))
    
    train_size = tf.data.experimental.cardinality(X_train).numpy()
    dev_size   = tf.data.experimental.cardinality(X_dev).numpy()
    test_size  = tf.data.experimental.cardinality(X_test).numpy()
    
    if minibatch_size == 0:
        minibatch_size = train_size
        
    nBatches = train_size//minibatch_size
    
    train_minibatches = train_dataset.batch(minibatch_size).prefetch(8)
    dev_minibatches   = dev_dataset.batch(dev_size).prefetch(8)
    test_minibatches  = test_dataset.batch(test_size).prefetch(8)
    
    # Training loop
    
    for epoch in range(num_epochs):
        
        train_cost = 0.
        
        for (minibatch_X, minibatch_Y) in train_minibatches:
            
            with tf.GradientTape() as tape:
                
                # Forward propagation, Loss computation
                parameters, y_hat_train = forward_propagation(tf.transpose(minibatch_X), parameters, layers)
                minibatch_cost = compute_cost(tf.transpose(minibatch_Y), y_hat_train, layers, tf.gather(minibatch_X, areaIdx, axis=1))
                
            
            trainable_variables = train_vars
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            train_cost += minibatch_cost
            
            mean, std = weight_statistics(parameters, layers, 'W')
            weight_mean.append(mean)
            weight_std.append(std)
            
            mean, std = weight_statistics(parameters, layers, 'Z')
            linear_mean.append(mean)
            linear_std.append(std)
            
            mean, std = grads_statistics(grads, layers)
            grad_mean.append(mean)
            grad_std.append(std)
            
        train_cost /= nBatches
            
        # Print the cost every 10 epochs
        if epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, train_cost))
            
            for (dev_minibatch_X, dev_minibatch_Y) in dev_minibatches:
            
                _, y_hat_dev = forward_propagation(tf.transpose(dev_minibatch_X),  parameters, layers)
                dev_cost  = compute_cost(tf.transpose(dev_minibatch_Y), y_hat_dev, layers, tf.gather(dev_minibatch_X, areaIdx, axis=1))
                
            print("Dev cost %f" %dev_cost)

            costs_plot.append([train_cost, dev_cost, epoch])
    
    N = 20
            
    weight_mean = convolve2d(weight_mean, np.ones((N,1))/N, boundary='symm')
    weight_std  = convolve2d(weight_std, np.ones((N,1))/N, boundary='symm')
            
    linear_mean = convolve2d(linear_mean, np.ones((N,1))/N, boundary='symm')
    linear_std  = convolve2d(linear_std, np.ones((N,1))/N, boundary='symm')
    
    grad_mean = convolve2d(grad_mean, np.ones((N,1))/N, boundary='symm')
    grad_std  = convolve2d(grad_std, np.ones((N,1))/N, boundary='symm')
    
    x = np.tile(np.arange(len(weight_mean)),(len(layers)-1,1)).T/nBatches
    
    plt.figure(1, figsize = (16,10))
    plt.suptitle('Moving average with window of size %d' %N)
    alpha = 0.45
    
    plt.subplot(2,3,1)
    plt.plot(x, weight_mean)
    plt.xlabel('# of Epochs')
    plt.ylabel('Mean value')
    plt.title('Weight')
    
    plt.subplot(2,3,4)
    plt.plot(x, weight_std)
    plt.xlabel('# of Epochs')
    plt.ylabel('Standard Deviation')
    
    plt.subplot(2,3,2)
    plt.plot(x, linear_mean)
    plt.xlabel('# of Epochs')
    plt.title('Linear function')
    
    plt.subplot(2,3,5)
    plt.plot(x, linear_std)
    plt.xlabel('# of Epochs')
    
    plt.subplot(2,3,3)
    plt.plot(x, grad_mean, alpha = alpha)
    plt.xlabel('# of Epochs')
    plt.title('Gradients of W')
    
    plt.subplot(2,3,6)
    plt.semilogy(x, grad_std, alpha = alpha)
    plt.xlabel('# of Epochs')
    plt.legend(['Layer ' + str(n+1) for n in range(len(layers))],frameon=False)
    
    # Verify and report the effectve dev and train cost
    
    train_dataset = tf.data.Dataset.zip((X_train, Y_train))
    dev_dataset   = tf.data.Dataset.zip((X_dev,   Y_dev))
    
    train_size = tf.data.experimental.cardinality(X_train).numpy()
    dev_size   = tf.data.experimental.cardinality(X_dev).numpy() 
    
    train_minibatches = train_dataset.batch(train_size).prefetch(8)
    dev_minibatches   = dev_dataset.batch(dev_size).prefetch(8)
    
    for (train_minibatch_X, train_minibatch_Y) in train_minibatches:
    
        _, y_hat_train = forward_propagation(tf.transpose(train_minibatch_X),  parameters, layers)
        train_cost  = compute_cost(tf.transpose(train_minibatch_Y), y_hat_train, layers, tf.gather(train_minibatch_X, areaIdx, axis=1))
            
    for (dev_minibatch_X, dev_minibatch_Y) in dev_minibatches:
    
        _, y_hat_dev = forward_propagation(tf.transpose(dev_minibatch_X),  parameters, layers)
        dev_cost  = compute_cost(tf.transpose(dev_minibatch_Y), y_hat_dev, layers, tf.gather(dev_minibatch_X, areaIdx, axis=1))
    
    return parameters, train_cost, dev_cost, costs_plot



