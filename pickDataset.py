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

class preprocess_features:

    def __init__(self, angles, resolutions, features):

        self.angles      = angles
        self.resolutions = resolutions
        self.features    = features

    def return_dataframe(self):
        
        DF = self.read_file(self.resolutions, self.angles)

        return DF[self.features]
    
    def read_file(self, resolutions, angles):
            
        data = []
        
        for res in resolutions:
        
            for ang in angles[res]:
                
                data.append(pd.read_csv(str('Features/' + resolutions[res] + str(ang))))
        
        return pd.concat(data, axis=0) 
        
##########################################################
########                 CODE BODY                ########
##########################################################

#angles     = {'LF': [0,10,20,30,40,50,60,70,80,90], 'HF': [10,30,50,70,90]}
angles     = {'LF': [0,10,20,30,40,50,60,70,80,90]}
resolution = {'LF': 'Coarsest'}
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

datasplit = preprocess_features(angles, resolution, variables)
DF = datasplit.return_dataframe()

minVals = DF.min().to_numpy()
maxVals = DF.max().to_numpy()

print(maxVals)
print(minVals)


















