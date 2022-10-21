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
    
        parameters['W' + str(l)] = tf.Variable(np.random.randn(layers[l], layers[l-1])/np.sqrt(layers[l-1]+layers[l]))
        parameters['b' + str(l)] = tf.Variable(np.random.randn(layers[l], 1))
        
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
        parameters['A' + str(l)] = tf.keras.activations.tanh(parameters['Z' + str(l)])
        
        
        assert(parameters['A' + str(l)].shape[0] == parameters['b' + str(l)].shape[0])
        assert(parameters['A' + str(l)].shape[1] == X.shape[1])
        assert(parameters['A' + str(l)].shape == parameters['Z' + str(l)].shape)
    
    parameters['Z' + str(L)] = tf.add(tf.matmul(parameters['W' + str(L)], parameters['A' + str(L-1)]), parameters['b' + str(L)])
    parameters['A' + str(L)] = tf.keras.activations.tanh(parameters['Z' + str(L)])
        
        
    assert(parameters['A' + str(L)].shape[0] == parameters['b' + str(L)].shape[0])
    assert(parameters['A' + str(L)].shape[1] == X.shape[1])
    assert(parameters['A' + str(L)].shape == parameters['Z' + str(L)].shape)
    
    
    return parameters


def compute_cost(labels, parameters, layers, area):
    
    L = len(layers)-1
    
    y_hat = parameters['A' + str(L)]
    
    ATimesSq = tf.multiply(tf.pow(tf.subtract(y_hat, labels),2), area)
    
    cost = tf.divide(tf.reduce_sum(ATimesSq),tf.reduce_sum(area)) 
    
    #ATimesSq = tf.sqrt( tf.reduce_mean(tf.pow(tf.subtract(y_hat, labels),2)))
    
    cost = cost
    
    return cost

def model(X_train, Y_train, X_dev, Y_dev, layers, learning_rate = 0.01,
          num_epochs = 3000, minibatch_size = 0, print_cost = True):
    
    # Initalize costs, variables, and optimizer
    
    parameters = initialize_parameters(layers)    
    
    train_vars = []
    costs = []
    
    for l in range(1,len(layers)):
        train_vars.append(parameters['W' + str(l)])
        train_vars.append(parameters['b' + str(l)])

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # Building training and dev set batches
    
    dataset = tf.data.Dataset.zip((X_train, Y_train))
    dev_dataset = tf.data.Dataset.zip((X_dev, Y_dev))
    
    train_size = tf.data.experimental.cardinality(X_train).numpy()
    dev_size   = tf.data.experimental.cardinality(X_dev).numpy()
    
    if minibatch_size == 0:
        minibatch_size = train_size
    
    minibatches = dataset.batch(minibatch_size).prefetch(8)
    dev_minibatches = dev_dataset.batch(dev_size).prefetch(8)
    
    #X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step    
    #Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster
    
    # Training loop
    
    for epoch in range(num_epochs):
        
        epoch_cost = 0.
        
        for (minibatch_X, minibatch_Y) in minibatches:
            
            with tf.GradientTape() as tape:
                
                # Forward propagation, Loss computation
                parameters = forward_propagation(tf.transpose(minibatch_X), parameters, layers)
                
                minibatch_cost = compute_cost(tf.transpose(minibatch_Y), parameters, layers, tf.gather(minibatch_X, layers[0]-1, axis=1))
                
            
            trainable_variables = train_vars
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost

        # Print the cost every 10 epochs
        if print_cost == True and epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            

            for l in range(1, len(layers)):
                print(tf.reduce_min(parameters['Z' + str(l)]).numpy(), tf.reduce_mean(parameters['Z' + str(l)]).numpy(), tf.reduce_max(parameters['Z' + str(l)]).numpy())
            #print ("Train accuracy %f" % train_accuracy)
            
            for (dev_minibatch_X, dev_minibatch_Y) in dev_minibatches:
            
                _ = forward_propagation(tf.transpose(dev_minibatch_X), parameters, layers)
                cost_dev  = compute_cost(tf.transpose(dev_minibatch_Y), parameters, layers, tf.gather(dev_minibatch_X, layers[0]-1, axis=1))
                
            print("Dev cost %f" %cost_dev)

            costs.append(epoch_cost)
            
    return parameters, costs



