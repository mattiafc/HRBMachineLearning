import numpy   as np
import pandas  as pd
import HRBProbes
import re
import torch
device = torch.device("cpu")

import itertools       as itt
from sklearn.decomposition   import PCA
from scipy.special           import rel_entr
from scipy.spatial.distance  import jensenshannon

from joblib         import Parallel, delayed

from statsmodels.nonparametric.kernel_density import KDEMultivariate

import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt

font = {'size'   : 15}

matplotlib.rc('font', **font)

def mmd(x, y, sigma):
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k = torch.exp((-1/(2*sigma**2)) * dists**2) + torch.eye(n+m)*1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY).numpy()

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



def splint_int_sets(HFAngles, dataFrame):
    
    positiveSet = HFAngles
    negativeSet = list(np.sort(list(set(range(0,100,10)) - set(HFAngles))))
    
    positiveDF = dataFrame.loc[dataFrame['theta'].isin(positiveSet)]
    negativeDF = dataFrame.loc[dataFrame['theta'].isin(negativeSet)]
    
    return positiveDF, negativeDF


def ndgrid(feature1, feature2):
    
    nElem = len(feature1)

    tempAll = np.zeros((1,2))

    for i in feature1:
        
        temp  = np.stack((np.ones((nElem,))*i,feature2),axis = 1)
        tempAll = np.vstack((tempAll,temp))
        
    return tempAll[1:,:] 


##############################
###### Proper functions ######
##############################


def compare_PCA(HFAngles, dataFrame, metric):
    
    positiveDF, negativeDF = splint_int_sets(HFAngles, dataFrame)

    #print(positiveDF.to_numpy().shape)

    pca = PCA(n_components=2)

    positivePCA = pca.fit_transform(positiveDF.to_numpy())
    negativePCA = pca.fit_transform(negativeDF.to_numpy())

    positiveKDE = KDEMultivariate(positivePCA, var_type = 'cc')
    negativeKDE = KDEMultivariate(negativePCA, var_type = 'cc')

    minimum = np.min(np.vstack((positivePCA, negativePCA)),axis = 0)
    maximum = np.max(np.vstack((positivePCA, negativePCA)),axis = 0)

    nElem = 50

    feature1 = np.linspace(minimum[0], maximum[0], num=nElem)
    feature2 = np.linspace(minimum[1], maximum[1], num=nElem)

    evalPoints = ndgrid(feature1, feature2)

    positivePDF = positiveKDE.pdf(evalPoints)
    negativePDF = negativeKDE.pdf(evalPoints)
    
    #metric = sum(rel_entr(positivePDF, negativePDF))
    if metric == 'JS':
        metric = jensenshannon(positivePDF,negativePDF)

        with open('datasetSearch.dat', 'a+') as out: 
            out.write('JS distance for set ' + str(np.asarray(HFAngles)) + ' is ' +str(metric) + '\n')
            
    elif metric == 'MMD':
        metric = MMD(positivePCA, negativePCA, 'multiscale')

        with open('datasetSearch.dat', 'a+') as out: 
            out.write('MMD for set ' + str(np.asarray(HFAngles)) + ' is ' +str(metric) + '\n')
    
    #elif metric == 'KL':
        #metric = sum(rel_entr(positivePDF, negativePDF))

        #with open('datasetSearch.dat', 'a+') as out: 
            #out.write('KL divergence for set ' + str(np.asarray(HFAngles)) + ' is ' +str(metric) + '\n')
        
    return
        
        
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

lista = list(itt.combinations([0,10,20,30,40,50,60,70,80,90],5))

#Parallel(n_jobs= 12)(delayed(compare_PCA)(sets, DF, 'MMD') for sets in lista[:len(lista)//2])

HFAngles   = []
similarity = []

with open('MMD.dat','r') as infile:
    for line in infile:
        temp = list(map(int,re.findall('[0-9]+', line)))
        similarity.append(float(re.findall("\d+\.\d+", line)[0]))
        HFAngles.append(list(map(int,re.findall('[0-9]+', line)))[:-2])
        
results = pd.DataFrame(list(zip(HFAngles, similarity,)), columns = ['Set', 'Similarity']).sort_values(by = 'Similarity', ascending = True)
results = results.drop_duplicates(subset=['Similarity'], keep = False)
print(results)




