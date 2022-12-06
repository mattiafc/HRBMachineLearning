import numpy as np
import re
import tensorflow as tf

with open("../MachineLearningOutput/ModelParameters/setup",'r') as infile:
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

parameters = {}

for l in range(1, nLayers+1):
    
    parameters['W' + str(l)] = tf.Variable(np.loadtxt("../MachineLearningOutput/ModelParameters/W"+str(l)+".csv", delimiter = ','))
    parameters['b' + str(l)] = tf.Variable(np.loadtxt("../MachineLearningOutput/ModelParameters/b"+str(l)+".csv", delimiter = ','))

return layers, learning_rate, epochs, minibatch, parameters