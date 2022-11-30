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

def initialize_parameters(layers_list):
    
    """
    Arguments:
    layers -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layers[l], layers[l-1])
                    bl -- bias vector of shape (layers[l], 1)
                  
    """
    
    parameters = {}

    L0 = 1

    for layers in layers_list:
        
        print(layers)    
        for l in range(1, len(layers)):
            
            idx = str(l+L0-1)
            
            parameters['W' + idx] = tf.Variable(np.random.randn(layers[l], layers[l-1])/np.sqrt(layers[l-1]))
            parameters['b' + idx] = tf.Variable(np.random.randn(layers[l], 1))
            print("Layer "+idx+" with shape " +str(parameters['W' + idx].shape))
            
            assert(parameters['W' + str(idx)].shape == (layers[l], layers[l - 1]))
            assert(parameters['b' + str(idx)].shape == (layers[l], 1))
            
        L0 += len(layers)-1

        
    return parameters

def standard_NN(X, parameters, layers, L0, output = 'linear'):
    
    A = tf.constant(X)
    
    L = len(layers)-1

    for l in range(1, L):
        
        idx = str(l+L0-1)
        
        parameters['Z' + idx] = tf.add(tf.matmul(parameters['W' + idx], A), parameters['b' + idx])
        A = tf.keras.activations.tanh(parameters['Z' + idx])
        
        assert(A.shape[0] == parameters['b' + idx].shape[0])
        assert(A.shape[1] == X.shape[1])
        assert(A.shape == parameters['Z' + idx].shape)
    
            
    idx = str(L+L0-1)
    parameters['Z' + idx] = tf.add(tf.matmul(parameters['W' + idx], A), parameters['b' + idx])
    
    if output == 'linear':
        A = parameters['Z' + idx]
    
    elif output == 'tanh':
        A = tf.keras.activations.tanh(parameters['Z' + idx])
    
    elif output == 'relu':
        A = tf.keras.activations.relu(parameters['Z' + idx])
    
    else:
        raise Exception("Output neuron activation %s has not been included" %output)
           
    assert(A.shape[0] == parameters['b' + idx].shape[0])
    assert(A.shape[1] == X.shape[1])
    assert(A.shape == parameters['Z' + idx].shape) 
    
    return parameters, A


def forward_propagation(X, parameters, layers_list):
    
    L0 = 1
    parameters, A_shared =   standard_NN(X, parameters, layers_list[0], L0, 'linear')
    
    #L0 += len(layers_list[0])-1
    #parameters, Cp1 = standard_NN(A_shared, parameters, layers_list[1], L0, 'linear')
    
    #L0 += len(layers_list[1])-1
    #parameters, Cp2 = standard_NN(A_shared, parameters, layers_list[2], L0, 'linear')
    
    #L0 += len(layers_list[2])-1
    #parameters, Cp3 = standard_NN(A_shared, parameters, layers_list[3], L0, 'linear')
    
    #L0 += len(layers_list[3])-1
    #parameters, Cp4 = standard_NN(A_shared, parameters, layers_list[4], L0, 'linear')
    
    #return parameters, tf.squeeze(tf.stack([Cp1, Cp2, Cp3, Cp4]))
    
    return parameters, A_shared
        

def compute_cost(X, labels, y_hat, area, scaling_factors, labelsIdx):
    
    LF_pred = tf.gather(X, labelsIdx, axis=0)
    
    rescaled_labels  = tf.add(tf.multiply(labels,scaling_factors[3]),scaling_factors[2])
    rescaled_NN_pred = tf.add(tf.multiply(y_hat,scaling_factors[3]),scaling_factors[2])
    rescaled_LF_pred = tf.add(tf.multiply(LF_pred,scaling_factors[1][labelsIdx]),scaling_factors[0][labelsIdx])    

    NN_RMSE = compute_RMSE(rescaled_labels, rescaled_NN_pred, area)
    LF_RMSE = compute_RMSE(rescaled_labels, rescaled_LF_pred, area)
    
    return NN_RMSE, LF_RMSE


def compute_RMSE(labels, y_hat, area):
    
    ATimesSq = tf.multiply(tf.pow(tf.subtract(y_hat, labels),2), area)
    
    RMSE = tf.sqrt(tf.divide(tf.reduce_sum(ATimesSq,1),tf.reduce_sum(area)))
    
    return RMSE

def model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, layers_list, areaIdx,
          learning_rate, num_epochs, minibatch_size, scaling_factors, labelsIdx):
    
    # Initalize costs, variables, and optimizer
    
    #weight_mean = []
    #weight_std  = []
    #linear_mean = []
    #linear_std  = []
    #grad_mean = []
    #grad_std  = []
    
    
    parameters = initialize_parameters(layers_list)
    
    train_vars = []
    costs_plot = []
    
    for l in range(1,sum([len(l) for l in layers_list])-len(layers_list)+1):
        
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
        #train_minibatches = train_dataset.shuffle(train_size, seed = epoch).batch(minibatch_size)
        
        for (minibatch_X, minibatch_Y) in train_minibatches:
            
            with tf.GradientTape() as tape:
                
                # Forward propagation, Loss computation
                parameters, y_hat_train = forward_propagation(tf.transpose(minibatch_X), parameters, layers_list)
                NN_RMSE, LF_RMSE = compute_cost(tf.transpose(minibatch_X), tf.transpose(minibatch_Y), y_hat_train, tf.gather(minibatch_X, areaIdx, axis=1), scaling_factors, labelsIdx)
                minibatch_cost = tf.divide(NN_RMSE,LF_RMSE)
                
            
            trainable_variables = train_vars
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            train_cost += minibatch_cost
            
            #mean, std = weight_statistics(parameters, layers, 'W')
            #weight_mean.append(mean)
            #weight_std.append(std)
            
            #mean, std = weight_statistics(parameters, layers, 'Z')
            #linear_mean.append(mean)
            #linear_std.append(std)
            
            #mean, std = grads_statistics(grads, layers)
            #grad_mean.append(mean)
            #grad_std.append(std)
            
        train_cost /= nBatches
            
        # Print the cost every 10 epochs
        if epoch % 10 == 0:
            print ("Cost after epoch "+str(epoch)+" : " + str(train_cost.numpy()))
            
            for (dev_minibatch_X, dev_minibatch_Y) in dev_minibatches:
            
                _, y_hat_dev = forward_propagation(tf.transpose(dev_minibatch_X),  parameters, layers_list)
                NN_RMSE, LF_RMSE  = compute_cost(tf.transpose(dev_minibatch_X), tf.transpose(dev_minibatch_Y), y_hat_dev, tf.gather(dev_minibatch_X, areaIdx, axis=1), scaling_factors, labelsIdx)
                dev_cost = tf.divide(NN_RMSE,LF_RMSE)
                
            print("Dev cost " +str(dev_cost.numpy()))

            costs_plot.append([train_cost, dev_cost, epoch])
    
    
    #for (test_minibatch_X, test_minibatch_Y) in test_minibatches:
            
        #_, y_hat_test = forward_propagation(tf.transpose(test_minibatch_X),  parameters, layers)
        #test_cost  = compute_cost(tf.transpose(test_minibatch_Y), y_hat_test, tf.gather(test_minibatch_X, areaIdx, axis=1), scaling_factors)
        
    #N = 20
            
    #weight_mean = convolve2d(weight_mean, np.ones((N,1))/N, boundary='symm')
    #weight_std  = convolve2d(weight_std, np.ones((N,1))/N, boundary='symm')
            
    #linear_mean = convolve2d(linear_mean, np.ones((N,1))/N, boundary='symm')
    #linear_std  = convolve2d(linear_std, np.ones((N,1))/N, boundary='symm')
    
    #grad_mean = convolve2d(grad_mean, np.ones((N,1))/N, boundary='symm')
    #grad_std  = convolve2d(grad_std, np.ones((N,1))/N, boundary='symm')
    
    #x = np.tile(np.arange(len(weight_mean)),(len(layers)-1,1)).T/nBatches
    
    #plt.figure(1, figsize = (16,10))
    #plt.suptitle('Moving average with window of size %d' %N)
    #alpha = 0.45
    
    #plt.subplot(2,3,1)
    #plt.plot(x, weight_mean)
    #plt.xlabel('# of Epochs')
    #plt.ylabel('Mean value')
    #plt.title('Weight')
    
    #plt.subplot(2,3,4)
    #plt.plot(x, weight_std)
    #plt.xlabel('# of Epochs')
    #plt.ylabel('Standard Deviation')
    
    #plt.subplot(2,3,2)
    #plt.plot(x, linear_mean)
    #plt.xlabel('# of Epochs')
    #plt.title('Linear function')
    
    #plt.subplot(2,3,5)
    #plt.plot(x, linear_std)
    #plt.xlabel('# of Epochs')
    
    #plt.subplot(2,3,3)
    #plt.plot(x, grad_mean, alpha = alpha)
    #plt.xlabel('# of Epochs')
    #plt.title('Gradients of W')
    
    #plt.subplot(2,3,6)
    #plt.semilogy(x, grad_std, alpha = alpha)
    #plt.xlabel('# of Epochs')
    #plt.legend(['Layer ' + str(n+1) for n in range(len(layers_list))],frameon=False)
    
    # Verify and report the effectve dev and train cost
    
    train_dataset = tf.data.Dataset.zip((X_train, Y_train))
    dev_dataset   = tf.data.Dataset.zip((X_dev,   Y_dev))
    
    train_size = tf.data.experimental.cardinality(X_train).numpy()
    dev_size   = tf.data.experimental.cardinality(X_dev).numpy() 
    
    train_minibatches = train_dataset.batch(train_size).prefetch(8)
    dev_minibatches   = dev_dataset.batch(dev_size).prefetch(8)
    
    for (train_minibatch_X, train_minibatch_Y) in train_minibatches:
    
        _, y_hat_train = forward_propagation(tf.transpose(train_minibatch_X),  parameters, layers_list)
        NN_RMSE, LF_RMSE  = compute_cost(tf.transpose(train_minibatch_X), tf.transpose(train_minibatch_Y), y_hat_train, tf.gather(train_minibatch_X, areaIdx, axis=1), scaling_factors, labelsIdx)
        train_cost = tf.divide(NN_RMSE,LF_RMSE)
            
    for (dev_minibatch_X, dev_minibatch_Y) in dev_minibatches:
    
        _, y_hat_dev = forward_propagation(tf.transpose(dev_minibatch_X),  parameters, layers_list)
        NN_RMSE, LF_RMSE  = compute_cost(tf.transpose(dev_minibatch_X), tf.transpose(dev_minibatch_Y), y_hat_dev, tf.gather(dev_minibatch_X, areaIdx, axis=1), scaling_factors, labelsIdx)
        dev_cost = tf.divide(NN_RMSE,LF_RMSE)
    
    return parameters, train_cost, dev_cost, costs_plot



