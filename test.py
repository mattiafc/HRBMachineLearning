import numpy as np

layers_list = [[11,15,14,13,12],[12,5,5,1],[12,4,1]]

parameters = {}

L0 = 1

#for layers in layers_list:
    
    #print(layers)    
    #for l in range(1, len(layers)):
        
        #idx = str(l+L0-1)
        
        #parameters['W' + idx] = np.random.randn(layers[l], layers[l-1])/np.sqrt(layers[l-1])
        #parameters['b' + idx] = np.random.randn(layers[l], 1)
        #print("Layer "+idx+" with shape " +str(parameters['W' + idx].shape))
        
        #assert(parameters['W' + str(idx)].shape == (layers[l], layers[l - 1]))
        #assert(parameters['b' + str(idx)].shape == (layers[l], 1))
        
    #L0 += len(layers)-1

layers_list = [[11,15,14,13,12],[12,5,5,1],[12,4,1]]

parameters = {}

L0 = 1

for layers in layers_list:
