import numpy as np
import HRBProbes
import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
import pandas as pd


#import matplotlib.pyplot as plt
#plt.style.use("ggplot")

def arrangeHeader(quantities):
    
    lista = quantities + ['coords']
    
    header = [None]*(len(lista)+2)
    
    for qty in lista:
        
        idx = featureNumber(qty)

        if isinstance(idx, int):
            header[idx] = qty
        else:
            header[idx[0]] = 'x'
            header[idx[1]] = 'y'
            header[idx[2]] = 'z'
    
    strHeader = ''
    for qty in header:
        strHeader += str(qty)
        strHeader += ','
        
    return strHeader[:-1]
        

def featureNumber(qty):
    
    if qty == 'coords':
        idx = [0,1,2]
            
    elif qty == 'TKE':
        idx = 3
        
    elif qty == 'U':
        idx = 4
        
    elif qty == 'gradP':
        idx = 5
        
    elif qty == 'UDotN':
        idx = 7
        
    elif qty == 'CfMean':
        idx = 6
        
    elif qty == 'meanCp':
        idx = 8
        
    elif qty == 'rmsCp':
        idx = 9
        
    elif qty == 'peakMaxCp':
        idx = 10
        
    elif qty == 'peakminCp':
        idx = 11
        
    elif qty == 'theta':
        idx = 12
        
    elif qty == 'LV0':
        idx = 13
        
    elif qty == 'Area':
        idx = 14
        
    return idx

def rotateNormals(normal, theta):
    
    cosine = np.cos(theta*np.pi/180)
    sine = np.sin(theta*np.pi/180)

    rotMat = np.matrix([[cosine, 0, -sine],[0, 1, 0],[sine, 0, cosine]])
    return(normal.dot(rotMat))


########################################
########################################
########################################


uStar = 0.4962
z0 = 0.0032
rho = 1.225
Uref = uStar/0.41*np.log(2.0/z0)

patches = {'F': 'front','R':'rear','L':'leeward','W':'windward','T':'top'}
normals = {'F': np.array([-1,0,0]),
           'R': np.array([ 1,0,0]),
           'L': np.array([0,0, 1]),
           'W': np.array([0,0,-1]),
           'T': np.array([0, 1,0])}

resolutions = {'Coarsest':'150001','Coarse':'240001'}
resolutions = {'Coarsest':'150001','Coarse':'240001'}

quantities = ['CfMean','TKE','U','gradP','UDotN','meanCp','rmsCp','peakMaxCp','peakminCp','theta','LV0','Area']
angles = list(range(0,100,10))
        
nColors = 25
color = 'coolwarm'
header = arrangeHeader(quantities)
print(header)
    
probes = {}

allCp = HRBProbes.readDat(patches, angles, resolutions, '../HRBPostProcessing/probesToDat/')
for ang in angles:
    
    for lvl in resolutions:
        plt.figure()

        nTaps = 0

        
        for pl in patches:

            dictKey = lvl + str(ang) + pl
            print(dictKey)
            faceNormal = rotateNormals(normals[pl],ang)
        
            data           = np.loadtxt(str('../HRBDataset/'+lvl+'/'+str(ang)+'/features' + str(lvl) + str(ang) + '/' +patches[pl]+'.00'+ resolutions[lvl] +'.pcd'),skiprows = 1)
            coords         = HRBProbes.rotateTranslateCoords(np.loadtxt(str('../HRBDataset/'+lvl+'/'+str(ang)+'/features' + str(lvl) + str(ang) + '/' +patches[pl]+'.pxyz'), skiprows = 1, usecols = (1,2,3)), ang)
            stressesVector = np.loadtxt(str('../HRBDataset/'+lvl+'/'+str(ang)+'/meanFieldStresses' + str(lvl) + str(ang) + '/' +patches[pl]+'.00'+ resolutions[lvl] +'.pcd'),skiprows = 1)
            
            srtrData, tList = HRBProbes.sortData(coords, patches[pl])
            nElem,_ = np.shape(data)
            
            if 'Cf' in quantities:
                wallStresses   = np.loadtxt(str('../HRBDataset/'+lvl+'/'+str(ang)+'/stresses' + str(lvl) + str(ang) + '/' +patches[pl]+'.highrise:tau_wall()_avg'),skiprows = -1)
                coordsStresses = HRBProbes.rotateTranslateCoords(np.loadtxt(str('../HRBDataset/'+lvl+'/'+str(ang)+'/stresses' + str(lvl) + str(ang) + '/' +patches[pl]+'.README'),skiprows = 4, usecols = (1,2,3)), ang)
                srtrStress, tList = HRBProbes.sortData(coordsStresses[-nElem:], patches[pl])
            else:
                srtrStress = srtrData
            
            nTaps+=nElem
            
            meanCp     = np.zeros(nElem)
            Cf     = np.zeros(nElem)
            CfMean = np.zeros(nElem)
            U      = np.zeros(nElem)
            UDotN  = np.zeros(nElem)
            TKE    = np.zeros(nElem)
            
            for qty in quantities:
                
                cont = 0
                
                if qty == 'TKE':
                    TKE = data[:,0]/(Uref*Uref)
                
                elif qty == 'U':
                    U = uStar/0.41*np.log(coords[:,1]/z0)/Uref
                
                elif qty == 'gradP':
                    gradP = data[:,1]*2.0/(0.5*rho*Uref*Uref)
                    
                elif qty == 'CfMean':
                    for stressVector in stressesVector:
                        stress   = stressVector.reshape(3,3)
                        CfMean[cont] = 1.79*pow(10,-5)*np.linalg.norm(stress.dot(faceNormal.T))/(0.5*rho*Uref*Uref)
                        cont += 1
                    
                elif qty == 'Cf':
                    Cf = np.array(wallStresses[-1:,-nElem:]/(0.5*rho*Uref*Uref)).reshape(-1)
                        
                elif qty == 'UDotN':
                    for cont in range(nElem):
                        UDotN[cont] = np.matrix([[1, 0, 0]]).dot(faceNormal.T)

                if lvl == 'Coarsest':
                    LV0 = 0.34

                elif lvl == 'Coarse':
                    LV0 = 0.23

                elif lvl == 'Fine':
                    LV0 = 0.15

            probes[dictKey] = {'TKE':TKE[srtrData],       'U':U[srtrData],         'gradP':gradP[srtrData], 
                               'CfMean': CfMean[srtrData], 'UDotN':UDotN[srtrData], 'Cf':Cf[srtrStress], 
                               'coords': coords[srtrData,:], 'theta': ang, 'LV0':LV0, 'Area':allCp[dictKey]['Area'],
                               'meanCp': allCp[dictKey]['meanCp'], 'peakMaxCp': allCp[dictKey]['peakMaxCp'], 
                               'rmsCp':  allCp[dictKey]['rmsCp'],  'peakminCp': allCp[dictKey]['peakminCp']}
        
        ############################################
        ### Assembling the CSV for the dataframe ###
        ############################################
        
        nFeatures = len(quantities)
        dataCSV   = np.zeros((nTaps,nFeatures + 3))
        total     = quantities + ['coords']
            
        cont = 0
        oldCont = 0
        
        for pl in patches:

            dictKey = lvl + str(ang) + pl
            
            cont += len(probes[dictKey]['meanCp'])

            for qty in total:
                
                dataCSV[oldCont:cont,featureNumber(qty)] = probes[dictKey][qty]
                
                
            oldCont = cont
        
        np.savetxt(str('Features/'+lvl + str(ang)), dataCSV, delimiter = ',', fmt='%1.12e', header = header)
        
        
        #for qty in quantities:

            
            #cbar = {'meanCp': [0, 0], 'rmsCp':[0,0], 'peakMaxCp':[0,0], 'peakminCp':[0,0],
                    #'TKE':[0, 0], 'U':[0, 0], 'gradP':[0, 0], 'CfMean':[0, 0], 'Cf':[0, 0], 'UDotN':[0,1]}
            
            #minVal, maxVal = HRBProbes.setColorbar(probes, [lvl], ang, patches, qty, cbar[qty], False)

            #CS = plt.tricontourf([0, 0, 0.0002, 0.002],[-0.0001, 0.0001, 0.0001, -0.0001], 
                                #[minVal[pl], 0.5*(minVal[pl]+maxVal[pl]), maxVal[pl], 0.5*(minVal[pl]+maxVal[pl])], 
                                #levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)
            #for pl in patches:

                #dictKey = lvl + str(ang) + pl
                
                ##print(probes[dictKey]['coords'])

                #if pl == 'T':
                    #plt.tricontourf(-probes[dictKey]['coords'][:,0]-0.35, probes[dictKey]['coords'][:,2]+2.35,  probes[dictKey][qty], 
                                        #levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)
                    

                #elif pl == 'L':
                    #plt.tricontourf(probes[dictKey]['coords'][:,0]+0.35, probes[dictKey]['coords'][:,1], probes[dictKey][qty], 
                                        #levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)

                #elif pl == 'W':
                    #plt.tricontourf(-0.35 - probes[dictKey]['coords'][:,0], probes[dictKey]['coords'][:,1], probes[dictKey][qty], 
                                        #levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)

                #elif pl == 'F':
                    #plt.tricontourf(probes[dictKey]['coords'][:,2], probes[dictKey]['coords'][:,1], probes[dictKey][qty], 
                                        #levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)

                #elif pl == 'R':
                    #plt.tricontourf(-1.6 - probes[dictKey]['coords'][:,2], probes[dictKey]['coords'][:,1], probes[dictKey][qty], 
                                        #levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)
                    
            #plt.title(str(qty+' for '+lvl+' LES, '+str(ang)+'deg'))
            #plt.colorbar(CS)
            #plt.savefig(str('./Plots/Features/' + qty + str(ang) + lvl + '.png'))
            ##plt.show()
            #plt.close()






