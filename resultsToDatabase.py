import numpy as np
import pandas as pd
import re

labels = ['meanCp','rmsCp','peakMaxCp','peakminCp']
sortSet      = 'Dev'
sortCriteria = 'Normalized'

fileName = str('../MachineLearningOutput/GridsearchUpdated.dat')

seed      = []
layers    = []
nLayers   = []
learning  = []
epochs    = []
minibatch = []
MFRMSE    = []
LFRMSE    = []

TrainRMSE_NN = []
DevRMSE_NN   = []
TrainRMSE_LF = []
DevRMSE_LF   = []

###################################
### Read results into dataframe ###
###################################

with open(fileName,'r') as infile:
    for line in infile:
        if 'Seed' in line:
            seed.append([int(x) for x in re.findall('[0-9]+', line)][0])
        if 'Layer' in line:
            layers.append(eval(line[22:]))
            nLayers.append(sum([len(l) for l in eval(line[22:])]))
        if 'Learning' in line:
            learning.append([float(x) for x in re.findall('[0-9].+', line)][0])
        if 'Number' in line:
            epochs.append( list(map(int,re.findall('[0-9]+', line)))[0] )
        if 'Minibatch' in line:
            minibatch.append([int(x) for x in re.findall('[0-9]+', line)][0])
        if 'MF NN' in line:
            MFRMSE.append([float(x) for x in re.findall(r'\d+\.\d+', line)])
        if 'LF LES' in line:
            LFRMSE.append([float(x) for x in re.findall(r'\d+\.\d+', line)])
        if 'Train' in line:
            TrainRMSE_NN.append([float(x) for x in re.findall('[0-9].\d+', line)][0:len(labels)])
            TrainRMSE_LF.append([float(x) for x in re.findall('[0-9].\d+', line)][len(labels):])
        if 'Dev' in line:
            DevRMSE_NN.append([float(x) for x in re.findall('[0-9].\d+', line)][0:len(labels)])
            DevRMSE_LF.append([float(x) for x in re.findall('[0-9].\d+', line)][len(labels):])
    
df = pd.DataFrame(list(zip(seed, layers, nLayers, learning, epochs, minibatch)), 
                columns = ['Seed', 'Layout', '# layers', 'Learn rate', 'Epochs', 'Batch size'])

cont = 0
for l in labels:
    
    df[str('MF RMSE '    + l)] = [data[cont] for data in MFRMSE]
    df[str('LF RMSE '    + l)] = [data[cont] for data in LFRMSE]
    df[str('Train RMSE NN ' + l)] = [data[cont] for data in TrainRMSE_NN]
    df[str('Train RMSE LF ' + l)] = [data[cont] for data in TrainRMSE_LF]
    df[str('Dev RMSE NN '   + l)] = [data[cont] for data in DevRMSE_NN]
    df[str('Dev RMSE LF '   + l)] = [data[cont] for data in DevRMSE_LF]
    
    cont += 1


###############################
### Sort results and output ###
###############################

columnsNN = []
columnsLF = []
RMSEList  = []

for l in labels:
    RMSEList.append([str(sortSet + ' RMSE NN ' + l)])
    columnsNN.append([str(sortSet + ' RMSE NN ' + l)])
    columnsLF.append([str(sortSet + ' RMSE LF ' + l)])

columnsNN.append(['Seed'])  
columnsLF.append(['Seed'])    
columnsNN = [elem for col in columnsNN for elem in col]
columnsLF = [elem for col in columnsLF for elem in col]

if sortCriteria == 'Normalized':

    dfNN = df[columnsNN]
    dfLF = df[columnsLF]

    output = pd.DataFrame(dfNN.values/dfLF.values, columns=dfNN.columns, index=dfNN.index)
    output['Seed'] = dfNN['Seed']

elif sortCriteria == 'Standard':
    output = df[columnsNN]
    
else:
    raise Exception('Ciucciello, sugalbot!!')

output['Metric'] = output.iloc[:,0:len(labels)].sum(axis=1)

output = output.sort_values(by = 'Metric', ascending = True)
    
print("===============================================================================================================")
print("Best setup sorted by %s set; %s mode" %(sortSet,sortCriteria))
temp = output.head(n=5)
print(temp)
print("\n===============================================================================================================")









