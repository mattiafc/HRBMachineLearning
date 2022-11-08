import numpy as np
import pandas as pd
import re

quantities = ['meanCp','rmsCp','peakMaxCp','peakminCp']
sortCriteria = 'Train RMSE'

for qty in quantities:

    fileName = str('../MachineLearningOutput/Definitive output/Gridsearch' + qty + '.dat')

    seed      = []
    layers    = []
    nLayers   = []
    learning  = []
    epochs    = []
    minibatch = []
    MFRMSE    = []
    LFRMSE    = []
    TrainRMSE = []
    DevRMSE   = []

    with open(fileName,'r') as infile:
        for line in infile:
            if 'Seed' in line:
                seed.append([int(x) for x in re.findall('[0-9]+', line)][0])
            if 'Layer' in line:
                layerSetup = list(map(int,re.findall('[0-9]+', line)))
                layers.append(layerSetup)
                nLayers.append(len(layerSetup)-2)
            if 'Learning' in line:
                learning.append([float(x) for x in re.findall('[0-9].+', line)][0])
            if 'Number' in line:
                epochs.append( list(map(int,re.findall('[0-9]+', line)))[0] )
            if 'Minibatch' in line:
                minibatch.append([int(x) for x in re.findall('[0-9]+', line)][0])
            if 'MF' in line:
                MFRMSE.append([float(x) for x in re.findall('[0-9].+', line)][0])
            if 'LF' in line:
                LFRMSE.append([float(x) for x in re.findall('[0-9].+', line)][0])
            if 'Train/dev' in line:
                TrainRMSE.append([float(x) for x in re.findall('[0-9].\d+', line)][0])
                DevRMSE.append([float(x) for x in re.findall('[0-9].\d+', line)][1])
        
    df = pd.DataFrame(list(zip(seed, layers, nLayers, learning, epochs, minibatch, MFRMSE, LFRMSE, TrainRMSE, DevRMSE)), 
                    columns = ['Seed', 'Layout', '# layers', 'Learn rate', 'Epochs', 'Batch size', 'MF RMSE', 'LF RMSE', 'Train RMSE', 'Dev RMSE'])
    
    df = df.sort_values(by = sortCriteria, ascending = True)
    
    print("===============================================================================================================")
    print("Best setup for " + str(qty) + ' sorted by ' + sortCriteria + '\n')
    temp = df.head(n=15)
    print(temp)
    print("\n===============================================================================================================")
