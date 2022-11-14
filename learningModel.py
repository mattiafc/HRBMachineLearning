import numpy   as np
import pandas  as pd
import tensorflow as tf
import tfNeuralNetGATTI as gatti
import HRBProbes
from datetime import datetime
from sklearn.model_selection import train_test_split

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
    
    def __init__(self, X_train_dev_nn, X_test_nn, y_train_dev, y_test, variables, label, learn_delta = False):
        
        #if dataset == ' Standard'
        
        self.areaIdx  = variables.index('Area')
        self.labelIdx = variables.index(label)
        
        self.learn_delta = learn_delta
        
        #if self.learn_delta == True:
            #y_train_dev = y_train_dev - X_train_dev_nn[self.labelIdx,:]
            #y_test      = y_train_dev - X_test_nn[self.labelIdx,:]
        
        self.nFeatures, self.nSamples = X_train_dev_nn.shape
        
        self.X_mean = np.mean(X_train_dev_nn, axis = 1, keepdims = True)
        self.X_std  = np.std(X_train_dev_nn,  axis = 1, keepdims = True, ddof = 1)+1e-10
        
        #self.y_mean = np.mean(y_train_dev, axis = 1, keepdims = True)
        #self.y_std  = np.std(y_train_dev,  axis = 1, keepdims = True, ddof = 1)+1e-10
        
        self.X_mean[self.areaIdx] = 0.0
        self.X_std[self.areaIdx]  = 1.0
        
        self.X_train_dev = (X_train_dev_nn - self.X_mean)/self.X_std
        self.y_train_dev = (y_train_dev - self.y_mean)/self.y_std
        
        self.X_test = (X_test_nn  - self.X_mean)/self.X_std
        self.y_test = (y_train_dev - self.y_mean)/self.y_std
        
        
        self.X_train, self.X_dev, self.y_train, self.y_dev = self.train_dev_split(self.X_train_dev.T, self.y_train_dev.T)
        
        #print('=====================================================')
        #print('Train set X shape: ' + str(self.X_train.shape))
        #print('Train set y shape: ' + str(self.y_train.shape))
        #print('Dev   set x shape: ' + str(self.X_dev.shape))
        #print('Dev   set y shape: ' + str(self.y_dev.shape))
        #print('X: ' + str(type(self.X_train)) + ', y: ' + str(type(self.y_train)))
        #print('Layer dimensions: ' + str(layers))
        #print('=====================================================\n')
        
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
                                        self.layers, self.areaIdx, learning_rate, num_epochs, minibatch_size)
        
        return self.parameters, train_cost, dev_cost, costs_plot
    
    ##########################################
    ###  Make predictions + evaluate RMSE  ###
    ##########################################
    
    def predictions_RMSE(self, test = False):
        
        # First: Compute the RMSE with respect to the various datasets
        
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
            
            for (X, y) in given_bacth:
                
                # Compute MF error (de-normalize data + RMSE computation)
                _, given_Cp_NN = gatti.forward_propagation(tf.transpose(X), self.parameters, self.layers)
                given_NN_RMSE  = gatti.compute_cost(tf.transpose(y), given_Cp_NN, self.layers, tf.gather(X, self.areaIdx, axis=1))
                
                # Compute LF error (de-normalize data + RMSE computation)
                given_y_LF     = tf.gather(X, self.labelIdx, axis=1)*self.X_std[self.labelIdx] + self.X_mean[self.labelIdx]
                given_LF_RMSE  = gatti.compute_cost(tf.transpose(y), (given_y_LF-self.y_mean)/self.y_std, 
                                            self.layers, tf.gather(X, self.areaIdx, axis=1))
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
        
        for (X, y) in pred_bacth:
            
            # Compute MF error (de-normalize data + RMSE computation)
            _, Cp_NN = gatti.forward_propagation(tf.transpose(X), self.parameters, self.layers)
            NN_RMSE  = gatti.compute_cost(tf.transpose(y), Cp_NN, self.layers, tf.gather(X, self.areaIdx, axis=1))
            
            # Compute LF error (de-normalize data + RMSE computation)
            y_LF = tf.gather(X, self.labelIdx, axis=1)*self.X_std[self.labelIdx] + self.X_mean[self.labelIdx]
            LF_RMSE  = gatti.compute_cost(tf.transpose(y), (y_LF-self.y_mean)/self.y_std, 
                                          self.layers, tf.gather(X, self.areaIdx, axis=1))
                
        print('=====================================================')
        print('RMSE comparison MF Neural Net vs LF LES over test_dev')
        print('RMSE consider the test set? ' + str(test))
        print('Train set MF RMSE %f:' %train_NN_RMSE)
        print('Train set LF RMSE %f:' %train_LF_RMSE)
        print('Dev   set MF RMSE %f:' %dev_NN_RMSE)
        print('Dev   set LF RMSE %f:' %dev_LF_RMSE)
        if test == True:
            print('Test  dev MF RMSE %f:' %dev_NN_RMSE)
            print('Test  dev LF RMSE %f:' %dev_LF_RMSE)
        print('The cumulative results for the sets above are:')   
        print("MF NN      RMSE %f" %NN_RMSE)
        print("LF LES     RMSE %f" %LF_RMSE)
        print('=====================================================')
        
        if test == True:
            return X_pred*self.X_std+self.X_mean, Cp_NN.numpy(), HF_Cp, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE, test_RMSE
        else:
            return X_pred*self.X_std+self.X_mean, Cp_NN.numpy(), HF_Cp, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE


    
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

    def split_dataset(self):
        
        if self.dataset == 'Standard':
            
            DF = self.read_file(self.resolutions, self.angles)

            X = DF[variables].values.T
            y = (DF[labels].values).reshape(1,-1)
            
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
            y_train = (trainLabels[labels].values).reshape(1,-1)

            X_test = testDF[variables].values.T
            y_test = (testLabels[labels].values).reshape(1,-1)
            
        
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
    
    
    
##########################################################
###### Grid search for hyperparameter optimization #######
##########################################################   



def parallelGridSearch(seed, X_train_dev, X_test, y_train_dev, y_test, variables, labels, test = False):

    np.random.seed(seed)
    
    #### Rms setup ###
    
    #n_hidden_layers = np.random.randint(1, high = 6)
    
    #layers      = (np.random.randint(15, size = int(n_hidden_layers))+2).tolist()
    #layers.insert(0, X_train_dev.shape[0])
    #layers.append(1)
    
    #learning_rate = 10**np.random.uniform(-5.0,-2.0)
    #n_epochs    = 701
    #batch_size    = int(2**np.round(np.random.uniform(4.0, 8.1)))
    
    n_hidden_layers = np.random.randint(1, high = 7)
    
    layers      = (np.random.randint(15, size = int(n_hidden_layers))+2).tolist()
    layers.insert(0, X_train_dev.shape[0])
    layers.append(1)
    
    learning_rate = 10**np.random.uniform(-5.0,-2.0)
    n_epochs      = 701
    batch_size    = int(2**np.round(np.random.uniform(4.0, 8.1)))

    neuralNet = neural_networks(X_train_dev, X_test, y_train_dev, y_test, variables, labels)

    parameters, train_cost, dev_cost, costs_plot = neuralNet.fit_neural_network(layers, learning_rate, n_epochs, batch_size)
    
    plt.savefig('../MachineLearningOutput/Plots/WeightAndGrad/Label:%s,Seed:%d.png' %(labels, seed))
    plt.close()
    
    if test == False:
        X_pred, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE  = neuralNet.predictions_RMSE(test)
    
    elif test == True:
        X_pred, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE, test_RMSE  = neuralNet.predictions_RMSE(test)

    trainDF = datasplit.read_file([resolution['LF']], angles['HF'])
    
    with open('../MachineLearningOutput/Gridsearch' + labels + '.dat', 'a+') as out:
        out.write('========================================\n')
        out.write('Seed number         : %d\n' %seed)
        out.write('Layer dimensions    : ' + str(layers)+'\n')
        out.write('Learning rate       : %f\n' %learning_rate)
        out.write('Number of epochs    : %d\n' %n_epochs)
        out.write('Minibatch size      : %d\n' %batch_size)
        out.write("MF NN  integral RMSE: %f\n" %NN_RMSE)
        out.write("LF LES integral RMSE: %f\n" %LF_RMSE)
        out.write("Train NN/LF     RMSE: %f, %f\n" %(train_RMSE[0], train_RMSE[1]))
        out.write("Dev   NN/LF     RMSE: %f, %f\n" %(dev_RMSE[0], dev_RMSE[1]))
        if test == True:
            out.write("Test  NN/LF     RMSE: %f, %f\n" %(test_RMSE[0], test_RMSE[1]))
        out.write('========================================\n')
    
    costs_plot = np.asarray(costs_plot)
    plt.plot(costs_plot[:,2], costs_plot[:,0], label = 'Train')
    plt.plot(costs_plot[:,2], costs_plot[:,1], label = 'Dev')
    plt.xlabel('# of Epochs')
    plt.ylabel('Cost')
    plt.title('Label: %s, Seed: %d' %(labels, seed))
    plt.legend(frameon=False)
    plt.savefig('../MachineLearningOutput/Plots/Costs/Label:%s,Seed:%d.png' %(labels, seed))
    plt.close()
    #plt.plot(np.squeeze(lastStep[-2:-1]-t0)*tStep*np.ones(y.shape),y,color='lime',linewidth = 2.0)
    
    return
    
    
    
##########################################################
######## Utilities to save and read dat for plots ########
##########################################################



def saveToDat(patches, angles, resolution, variables, labels, trainDF, X_pred, Cp_NN, Cp_HF):

    predDF = pd.DataFrame(trainDF, columns = variables)

    predDF[str(labels) + 'NN'] = Cp_NN.T
    predDF[str(labels) + 'HF'] = Cp_HF.T
    cumulativeDF = pd.merge(trainDF,predDF)

    NNEntries = ['# x', 'y', 'z', labels + 'NN']
    HFEntries = ['# x', 'y', 'z', labels + 'HF']

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
            
            NNPred = df[NNEntries].values
            HFPred = df[HFEntries].values
            
            fNameNN = str('../MachineLearningOutput/NNPred/NeuralNet' + str(ang) + pl + '.dat')
            fNameHF = str('../MachineLearningOutput/HFPred/' + resolution['HF'] + str(ang) + pl + '.dat')
            
            np.savetxt(fNameNN, NNPred, header = str(NNEntries))
            np.savetxt(fNameHF, HFPred, header = str(HFEntries))
    
def readDat(patches, angles, deltas, directory = '../MachineLearningOutput/probesToDat'):

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
                
                probes[dictKey] = {'coords': coords, 'rmsCp': np.around(temp[:,3],roundoff)}

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

    #X_pred, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE  = neuralNet.predictions_RMSE(False)
    
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


##########################################################
########                 CODE BODY                ########
##########################################################

#angles     = {'LF': [0,10,20,30,40,50,60,70,80,90], 'HF': [10,30,50,70,90]}
angles     = {'LF': [0,10,20,30,40,50,60,70,80,90], 'HF': [0,20,40,60,80]}
resolution = {'LF': 'Coarsest', 'HF': 'Coarse'}
variables  = ['CfMean','TKE','U','gradP','UDotN','theta','meanCp','rmsCp','peakminCp','peakMaxCp','Area']
labels     = 'meanCp'

patches = {'F':'front','L':'leeward','R':'rear','T':'top','W':'windward'}

CpScale = {'0' :{'meanCp': [-1.2, 1.0],'rmsCp': [0, 0.36],'peakMaxCp': [-0.5, 1.7],'peakminCp': [-2.5, 0.5]},
           '10':{'meanCp': [-1.2, 1.0],'rmsCp': [0, 0.36],'peakMaxCp': [-0.5, 1.7],'peakminCp': [-2.5, 0.5]},
           '20':{'meanCp': [-1.2, 1.0],'rmsCp': [0, 0.36],'peakMaxCp': [-0.5 ,1.7],'peakminCp': [-2.5, 0.5]},
           '30':{'meanCp': [-1.5, 1.0],'rmsCp': [0, 0.45],'peakMaxCp': [-0.5, 1.7],'peakminCp': [-3.0, 0.5]},
           '40':{'meanCp': [-2.0, 1.0],'rmsCp': [0, 0.55],'peakMaxCp': [-1.0, 1.5],'peakminCp': [-4.0, 0.3]},
           '50':{'meanCp': [-2.0, 1.0],'rmsCp': [0, 0.55],'peakMaxCp': [-1.0, 1.7],'peakminCp': [-4.0, 0.3]},
           '60':{'meanCp': [-1.2, 1.0],'rmsCp': [0, 0.45],'peakMaxCp': [-0.5, 1.5],'peakminCp': [-3.5, 0.3]},
           '70':{'meanCp': [-1.2, 1.0],'rmsCp': [0, 0.45],'peakMaxCp': [-0.5, 1.5],'peakminCp': [-3.0, 0.5]},
           '80':{'meanCp': [-1.0, 1.0],'rmsCp': [0, 0.35],'peakMaxCp': [-0.3, 1.5],'peakminCp': [-2.5, 0.5]},
           '90':{'meanCp': [-1.0, 1.0],'rmsCp': [0, 0.30],'peakMaxCp': [-0.3 ,1.5],'peakminCp': [-2.0, 0.5]}}

datasplit = preprocess_features(angles, resolution, variables, labels, 'MultiFidelity')

X_train_dev, X_test, y_train_dev, y_test = datasplit.split_dataset()

    
#with open('../MachineLearningOutput/Gridsearch' + labels + '.dat', 'a+') as out:
    #now = datetime.now()
    #out.write('\n'*10+'Gridsearch performed on ' + str(now.strftime("%d %m %Y, %H:%M:%S"))+ '\n'*10)
    
#_ = Parallel(n_jobs= 1)(delayed(parallelGridSearch)(seed, X_train_dev, X_test, y_train_dev, y_test, variables, labels)
                            #for seed in range(435,1000))

#bounds = [(-2,-4),(4.51,7.49),(3.51,16.49), (3.51,8.49)]
#res = optimize.differential_evolution(optimizer_wrap, bounds, args = (X_train_dev, X_test, y_train_dev, y_test, variables, labels),
                                      #popsize = 24, seed = 4, workers = 12, integrality = [False, True, True, True])
#print(res)

test = True

neuralNet = neural_networks(X_train_dev, X_test, y_train_dev, y_test, variables, labels)
parameters, train_cost, dev_cost, costs_plot = neuralNet.fit_neural_network([11, 12, 15, 14, 1], 0.00122, 701, 128)
plt.close()

X_pred, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE, test_RMSE  = neuralNet.predictions_RMSE(test)
    
if test == False:
    X_pred, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE  = neuralNet.predictions_RMSE(test)

if test == True:
    X_pred, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE, train_RMSE, dev_RMSE, test_RMSE  = neuralNet.predictions_RMSE(test)

trainDF = datasplit.read_file([resolution['LF']], angles['HF'])

print('========================================')
print("MF NN  integral RMSE: %f" %NN_RMSE)
print("LF LES integral RMSE: %f" %LF_RMSE)
print("Train NN/LF     RMSE: %f, %f" %(train_RMSE[0], train_RMSE[1]))
print("Dev   NN/LF     RMSE: %f, %f" %(dev_RMSE[0], dev_RMSE[1]))
if test == True:
    print("Test    NN/LF     RMSE: %f, %f" %(test_RMSE[0], test_RMSE[1]))
print('========================================')

#X_pred, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE  = neuralNet.predictions_RMSE(True)

#np.random.seed(seed)

#n_hidden_layers = np.round(np.random.uniform(1.0,6.1))

#layers      = (np.random.randint(12, size = int(n_hidden_layers))+2).tolist()
#layers.insert(0, X_train_dev.shape[0])
#layers.append(1)

#learning_rate = 10**np.random.uniform(-5.0,-2.0)
#n_epochs    = 11
##batch_size    = int(2**np.round(np.random.uniform(4.0, 8.1)))
#batch_size    = 256

#neuralNet = neural_networks(X_train_dev, X_test, y_train_dev, y_test, variables, labels)

#parameters, train_cost, dev_cost, costs_plot = neuralNet.fit_neural_network(layers, learning_rate, n_epochs, batch_size)

#X_pred, Cp_NN, Cp_HF, NN_RMSE, LF_RMSE  = neuralNet.predictions_RMSE(False)

#trainDF = datasplit.read_file([resolution['LF']], angles['HF'])

#saveToDat(patches, angles['HF'], resolution, variables, labels, trainDF, X_pred, Cp_NN, Cp_HF)

##Neural nets has been fitted, used to predict, and the output has been saved into a .dat 
     
#for ang in angles['HF']:

    #probes = readDat(patches, [ang], ['NeuralNet'], directory = '../MachineLearningOutput/NNPred/')
    #HRBProbes.plotQty(probes, ['NeuralNet'], [ang], patches, [labels], CpScale[str(ang)], directory = '../MachineLearningOutput/Plots/', resCompare = False)

    #probes = readDat(patches, [ang], [resolution['HF']], directory = 'HFPred/')
    #HRBProbes.plotQty(probes, [resolution['HF']], [ang], patches, [labels], CpScale[str(ang)], directory = '../MachineLearningOutput/Plots/', resCompare = False)










