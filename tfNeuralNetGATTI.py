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
        parameters['b' + str(l)] = tf.Variable(np.random.randn(layers[l], 1))
        #parameters['b' + str(l)] = tf.Variable(np.zeros((layers[l], 1)))
        
        #assert(parameters['W' + str(l)].shape == (layers[l], layers[l - 1]))
        #assert(parameters['b' + str(l)].shape == (layers[l], 1))

        
    return parameters

def forward_propagation(X, parameters, layers):
    
    A = tf.constant(X)
    
    L = len(layers)-1

    for l in range(1, L):
        
        Z = tf.add(tf.matmul(parameters['W' + str(l)], A), parameters['b' + str(l)])
        A = tf.keras.activations.tanh(Z)
        
        
        #assert(A.shape[0] == parameters['b' + str(l)].shape[0])
        #assert(A.shape[1] == X.shape[1])
        #assert(A.shape == Z.shape)
    
    Z = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])
    A = tf.keras.activations.tanh(Z)
        
        
    #assert(A.shape[0] == parameters['b' + str(L)].shape[0])
    #assert(A.shape[1] == X.shape[1])
    #assert(A.shape == Z.shape)
    
    
    return parameters, A


def compute_cost(labels, y_hat, layers, area):
    
    ATimesSq = tf.multiply(tf.pow(tf.subtract(y_hat, labels),2), area)
    
    RMSE = tf.sqrt(tf.divide(tf.reduce_sum(ATimesSq),tf.reduce_sum(area)))
    
    return RMSE

def model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, layers, areaIdx,
          learning_rate, num_epochs, minibatch_size, test_check = False):
    
    # Initalize costs, variables, and optimizer
    
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
    
    #X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step    
    #Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster
    
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
            
        train_cost /= nBatches
        
        
        # Print the cost every 10 epochs
        if epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, train_cost))
            
            for (dev_minibatch_X, dev_minibatch_Y) in dev_minibatches:
            
                _, y_hat_dev = forward_propagation(tf.transpose(dev_minibatch_X),  parameters, layers)
                dev_cost  = compute_cost(tf.transpose(dev_minibatch_Y), y_hat_dev, layers, tf.gather(dev_minibatch_X, areaIdx, axis=1))
                
            print("Dev cost %f" %dev_cost)

            costs_plot.append([train_cost, dev_cost, epoch])
    
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



