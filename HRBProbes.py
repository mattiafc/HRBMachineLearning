import copy
import sys
import numpy           as np
import multiprocessing as mp

from joblib         import Parallel, delayed
#from scipy.signal   import welch, hanning
from scipy.optimize import curve_fit
from scipy.stats    import pearsonr


import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt
#plt.style.use("ggplot")

font = {'size'   : 20}

matplotlib.rc('font', **font)

def writeDelta(patches, angles, LF, HF, correction, k = 5):
    
    probes = {}
    
    for ang in angles:
        for pl in patches:
            
            if ang<0:
                ang = 360+ang
            if ang >360:
                ang = ang-360
                
            iterable = [LF, HF]
            acronym = ''
            
            for resolution in iterable:
                if resolution == 'Coarsest':
                    acronym+='CC'
                elif resolution == 'Coarse':
                    acronym+='C'
                elif resolution == 'Fine':
                    acronym+='F'
                elif resolution == 'Finest':
                    acronym+='FF'
                    
            acronym += correction
                    
            LFKey = LF + str(ang) + pl
            HFKey = HF + str(ang) + pl
            dictKey = str(acronym) + str(ang) + pl
            
            LFData = readDat({pl: patches[pl]}, [ang], {LF: 0.0008}, 'probesToDat/')
            HFData = readDat({pl: patches[pl]}, [ang], {HF: 0.0005}, 'probesToDat/')
            
            fName = str('probesToDat/'+dictKey+'.dat')
            
            if 'sum' in acronym:
                probes[dictKey] = {'coords':   HFData[HFKey]['coords']
                                , 'meanCp':    HFData[HFKey]['meanCp']    - LFData[LFKey]['meanCp']
                                , 'rmsCp':     HFData[HFKey]['rmsCp']     - LFData[LFKey]['rmsCp']
                                , 'peakminCp': HFData[HFKey]['peakminCp'] - LFData[LFKey]['peakminCp']
                                , 'peakMaxCp': HFData[HFKey]['peakMaxCp'] - LFData[LFKey]['peakMaxCp']
                                , 'Area': HFData[HFKey]['Area']}
            
            elif 'dot' in acronym:
                probes[dictKey] = {'coords':   HFData[HFKey]['coords']
                                , 'meanCp':    HFData[HFKey]['meanCp']    / LFData[LFKey]['meanCp']
                                , 'rmsCp':     HFData[HFKey]['rmsCp']     / LFData[LFKey]['rmsCp']
                                , 'peakminCp': HFData[HFKey]['peakminCp'] / LFData[LFKey]['peakminCp']
                                , 'peakMaxCp': HFData[HFKey]['peakMaxCp'] / LFData[LFKey]['peakMaxCp']
                                , 'Area': HFData[HFKey]['Area']}
            
            elif 'skewDot' in acronym:
                probes[dictKey] = {'coords':   HFData[HFKey]['coords']
                                , 'meanCp':    (HFData[HFKey]['meanCp']   + k) / (LFData[LFKey]['meanCp']   + k)
                                , 'rmsCp':     (HFData[HFKey]['rmsCp']    + k) / (LFData[LFKey]['rmsCp']    + k)
                                , 'peakminCp': (HFData[HFKey]['peakminCp']+ k) / (LFData[LFKey]['peakminCp']+ k)
                                , 'peakMaxCp': (HFData[HFKey]['peakMaxCp']+ k) / (LFData[LFKey]['peakMaxCp']+ k)
                                , 'Area': HFData[HFKey]['Area']}
            else:
                print('UNDEFINED CORRECTION!!!')
                sys.exit()
                
            temp, tList = sortData(LFData[LFKey]['coords'], patches[pl])
            
            np.savetxt(str('probesToDat/'+dictKey+'.dat') ,np.transpose([probes[dictKey]['coords'][:,0], probes[dictKey]['coords'][:,1], probes[dictKey]['coords'][:,2], 
                    probes[dictKey]['meanCp'], probes[dictKey]['rmsCp'], probes[dictKey]['peakminCp'], probes[dictKey]['peakMaxCp'],tList[temp],LFData[LFKey]['Area']]),
                    header = 'x'+' '*17+'y' +' '*17+'z'+' '*17+'meanCp'+' '*13+'rmsCp'+' '*13+'peakminCp'+' '*9+'peakMaxCp'+' '*9+'tapCode'+' Area',
                    fmt = '%1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %d   %1.12f')

    return probes


def sortData(coordsFromProbes, patch):

    coordsCode = np.loadtxt('probesLocation/' + patch + 'CodedArea', skiprows=1, usecols = (0,1,2))
    tapCode = np.loadtxt('probesLocation/' + patch + 'CodedArea', skiprows=1, usecols = 3)
    coordsToSort = np.zeros((len(tapCode),4))
    cont = 0
    for tris in coordsFromProbes:
        diffNorm = np.linalg.norm(coordsCode-tris,axis = 1)
        idx = np.argmin(diffNorm)
        coordsToSort[cont] = [tris[0],tris[1],tris[2],tapCode[idx]]
        cont +=1
    sorter = np.argsort(coordsToSort[:,3])
    
    return sorter, coordsToSort[:,3]

def readPressure(counter, directory, patch):
    
    filename = directory + '/' + patch + '.' + str(counter).zfill(8) + '.pcd'
    return np.loadtxt(filename, skiprows=1)

def rotateTranslateCoords(coords, angle):
    
    theta = -angle*np.pi/180
    xmin = 0;
    xMax = 1.0;
    zmin = -0.15;
    zMax = 0.15;

    nPoints = np.shape(coords)[0]
    newCoords = np.zeros(np.shape(coords))

    P10 = np.array([[xmin],[2],[zmin]])
    P20 = np.array([[xMax],[2],[zmin]])
    P30 = np.array([[xMax],[2],[zMax]])
    P40 = np.array([[xmin],[2],[zMax]])
    
    cosine = np.cos(theta)
    sine = np.sin(theta)
    
    rotMat = np.matrix([[cosine, 0, -sine],[0, 1, 0],[sine, 0, cosine]])
    
    P1 = rotMat*P10;
    P2 = rotMat*P20;
    P3 = rotMat*P30;
    P4 = rotMat*P40;

    displacement =  0.5*(np.min([P1[2], P2[2], P3[2], P4[2]]) +  np.max([P1[2], P2[2], P3[2], P4[2]]));

    rotMat = np.matrix([[cosine, 0, sine],[0, 1, 0],[-sine, 0, cosine]])
    #coords[:,2] = coords[:,2] + displacement
    
    for i in range(nPoints):
        temp = np.matrix([[coords[i,0]],[coords[i,1]],[coords[i,2]+displacement]])
        newCoords[i,:] = (rotMat*temp).transpose()

    return newCoords

def curve(x, m, q):
    return m*x+q

def peakPressure(pressure, POE, windowTime, dt, minOrMax):
    
    nTaps, nSamples = np.shape(pressure)
    windowSize = int(np.floor(windowTime/dt))
    nWindows = int(np.floor(nSamples/windowSize))
    indices = np.array([i*windowSize for i in range(nWindows+1)])
    
    peakCp = np.zeros(nTaps)
    if nWindows >=2:
        for tap in range(nTaps):
            
            peakCollect = np.zeros(nWindows)
            for i in range(len(indices)-1):
                
                if minOrMax == 'min':
                    peakCollect[i] = np.min(pressure[tap,indices[i]:indices[i+1]])
                elif minOrMax == 'Max':
                    peakCollect[i] = np.max(pressure[tap,indices[i]:indices[i+1]]) 
                else:
                    raise Exception("It's either min or Max")
            
                peaks = np.sort(peakCollect)[::-1]
                piecewiseLogCDF = -np.log(-np.log(np.arange(1,nWindows+1)/(nWindows+1)))
            
                fitParams, covParams = curve_fit(curve, piecewiseLogCDF, peaks)
                peakCp[tap] = (fitParams[1] - fitParams[0]*np.log(-np.log(1-POE)))
            
    return peakCp


def setColorbar(probes, deltas, ang, patches, qty, cbar, resCompare):
    
                
    minVal = {}
    maxVal = {}
    if resCompare:
        for pl in patches:
            
            concatenateQty = []
            for lvl in deltas:
                
                dictKey = lvl + str(int(ang)) + pl
                concatenateQty = np.append(concatenateQty, probes[dictKey][qty])
                
            minVal.update({pl: np.min(concatenateQty)})
            maxVal.update({pl: np.max(concatenateQty)})
                
                
    else:
        
        minVal = {}
        maxVal = {}
        
        if cbar[0] == cbar[1]:
            
            concatenateQty = []
            for pl in patches:
                for lvl in deltas:
                    #print(probes)
                    dictKey = lvl + str(int(ang)) + pl
                    concatenateQty = np.append(concatenateQty, probes[dictKey][qty])
            
            for pl in patches:
                minVal.update({pl: np.min(concatenateQty)})
                maxVal.update({pl: np.max(concatenateQty)})
                
        else:
            
            for pl in patches:
                minVal.update({pl: cbar[0]})
                maxVal.update({pl: cbar[1]})
        
        
    return minVal, maxVal


def plotQty(probes, deltas, angles, patches, quantities, cbarDict, directory = './Plots/', resCompare = False):
    
    nColors = 25
    #my_dpi = 100
    
    #concatenateQty = []
    
                
    #plt.figure(figsize=(2560/my_dpi, 480/my_dpi), dpi=my_dpi)
    for qty in quantities:
        
        cbar = cbarDict[qty]
        
        for lvl in deltas:
            for ang in angles:
                
                minVal, maxVal = setColorbar(probes, deltas, ang, patches, qty, cbar, resCompare)
                
                color = 'coolwarm'
                
                
                ## THIS PLOT IS ONLY REQUIRED TO CORRECTLY VISUALIZE THE COLORBAR ##
                ## IN FACT, IT'S A SUPERSMALL SQUARE WITH CONTOUR RANGE FROM MIN TO MAX##
                
                stringaOld = str(qty + ' for ' + str(ang) + 'Deg')

                #if 'meanCp' in stringaOld :
                    #stringa = 'Mean Cp'
                    #plt.subplot(141)

                #elif 'rmsCp' in stringaOld :
                    #stringa = 'Rms Cp'
                    #plt.subplot(142)

                #elif 'peakMaxCp' in stringaOld :
                    #stringa = 'Peak Max Cp'
                    #plt.subplot(143)

                #elif 'peakminCp' in stringaOld :
                    #stringa = 'Peak Min Cp'
                    #plt.subplot(144)
                
                if not(resCompare):
                    for pl in patches:
                        CS = plt.tricontourf([0, 0, 0.0002, 0.002],[-0.0001, 0.0001, 0.0001, -0.0001], 
                                            [minVal[pl], 0.5*(minVal[pl]+maxVal[pl]), maxVal[pl], 0.5*(minVal[pl]+maxVal[pl])], 
                                            levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)
                
                for pl in patches:
                    
                    dictKey = lvl + str(int(ang)) + pl
                    
                    if pl == 'T':
                        plt.tricontourf(-probes[dictKey]['coords'][:,0]-0.35, probes[dictKey]['coords'][:,2]+2.35,  probes[dictKey][qty], 
                                        levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)
                        
                    
                    elif pl == 'L':
                        plt.tricontourf(probes[dictKey]['coords'][:,0]+0.35, probes[dictKey]['coords'][:,1], probes[dictKey][qty],
                                        levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)
                    
                    elif pl == 'W':
                        plt.tricontourf(-0.35 - probes[dictKey]['coords'][:,0], probes[dictKey]['coords'][:,1], probes[dictKey][qty],
                                        levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)
                    
                    elif pl == 'F':
                        plt.tricontourf(probes[dictKey]['coords'][:,2], probes[dictKey]['coords'][:,1], probes[dictKey][qty], 
                                        levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)
                    
                    elif pl == 'R':
                        plt.tricontourf(-1.6 - probes[dictKey]['coords'][:,2], probes[dictKey]['coords'][:,1], probes[dictKey][qty],
                                        levels = nColors, vmin = minVal[pl], vmax = maxVal[pl], cmap = color)
                
                    plt.axis('equal')
                    #if resCompare:
                        #plt.colorbar()
                            
                if not(resCompare):
                    plt.colorbar(CS)
                    #plt.colorbar(CS, ticks = np.linspace(cbar[0],cbar[1],3))
                    
                    plt.title(directory+str(int(ang))+lvl)
                    plt.axis('off')
                        
                plt.savefig(directory+str(int(ang))+lvl + '.png', bbox_inches='tight')
                #plt.show(block=True)
                plt.close('all')
                plt.close()
    
    return

def makeCasePeriodic(patches, angles, deltas, t0, tf):
            
    probes = {}
    
    for lvl in deltas:
        for ang in angles:
            for pl in patches:
                
                patchOrig = {}
                #print(dictKey)
                
                if ang<0:
                    ang = 360+ang
                #print(ang)
                dictKey  = lvl + str(ang) + pl
                
                if ang >= 0 and ang <= 90:
                    angOrig = int(ang)
                    plOrig = copy.deepcopy(pl)
                    
                elif ang > 90 and ang <= 180:
                    angOrig = int(180-ang)
                    if pl == 'L' or pl == 'W' or pl == 'T':
                            plOrig = copy.deepcopy(pl)
                    elif pl == 'R':
                            plOrig = 'F'
                    elif pl == 'F':
                            plOrig = 'R'
                            
                            
                elif ang > 180 and ang <= 270:
                    angOrig = int(ang-180)
                    if pl == 'T':
                            plOrig = copy.deepcopy(pl)
                    elif pl == 'R':
                            plOrig = 'F'
                    elif pl == 'F':
                            plOrig = 'R'
                    elif pl == 'L':
                            plOrig = 'W'
                    elif pl == 'W':
                            plOrig = 'L'
                    
                elif ang > 270 and ang <= 360:
                    angOrig = int(360-ang)
                    if pl == 'R' or pl == 'F' or pl == 'T':
                            plOrig = copy.deepcopy(pl)
                    elif pl == 'L':
                            plOrig = 'W'
                    elif pl == 'W':
                            plOrig = 'L'
                
                patchOrig[pl] = patches[plOrig]    
                dictKeyOrig = lvl + str(angOrig) + plOrig
                
                probesOrig = readDat(patches, [angOrig], deltas, directory = 'probesToDat/')
                
                
                if ang >= 0 and ang <= 90:
                    newCoords = np.transpose(np.array([probesOrig[dictKeyOrig]['coords'][:,0],probesOrig[dictKeyOrig]['coords'][:,1],probesOrig[dictKeyOrig]['coords'][:,2]]))
                    
                if ang > 90 and ang <= 180:
                    if pl == 'W'or pl == 'L' or pl == 'T':
                        newCoords = np.transpose(np.array([1-probesOrig[dictKeyOrig]['coords'][:,0],probesOrig[dictKeyOrig]['coords'][:,1],probesOrig[dictKeyOrig]['coords'][:,2]]))
                    if pl == 'R'or pl == 'F':
                        newCoords = np.transpose(np.array([probesOrig[dictKeyOrig]['coords'][:,0],probesOrig[dictKeyOrig]['coords'][:,1],probesOrig[dictKeyOrig]['coords'][:,2]]))
                        
                elif ang > 180 and ang <= 270:
                    if pl == 'W'or pl == 'L':
                        newCoords = np.transpose(np.array([1-probesOrig[dictKeyOrig]['coords'][:,0],probesOrig[dictKeyOrig]['coords'][:,1],probesOrig[dictKeyOrig]['coords'][:,2]]))
                    elif pl == 'R'or pl == 'F':
                        newCoords = np.transpose(np.array([probesOrig[dictKeyOrig]['coords'][:,0],probesOrig[dictKeyOrig]['coords'][:,1],-probesOrig[dictKeyOrig]['coords'][:,2]]))
                    elif pl == 'T':
                        newCoords = np.transpose(np.array([1-probesOrig[dictKeyOrig]['coords'][:,0],probesOrig[dictKeyOrig]['coords'][:,1],-probesOrig[dictKeyOrig]['coords'][:,2]]))
                    
                elif ang > 270 and ang <= 360:
                    if pl == 'R'or pl == 'F' or pl == 'T':
                        newCoords = np.transpose(np.array([probesOrig[dictKeyOrig]['coords'][:,0],probesOrig[dictKeyOrig]['coords'][:,1],-probesOrig[dictKeyOrig]['coords'][:,2]]))
                    if pl == 'L'or pl == 'W':
                        newCoords = np.transpose(np.array([probesOrig[dictKeyOrig]['coords'][:,0],probesOrig[dictKeyOrig]['coords'][:,1],probesOrig[dictKeyOrig]['coords'][:,2]]))
                
                print(dictKey)
                    
                srtr, tList = sortData(newCoords, patches[pl])
                #area = np.loadtxt('probesLocation/'+patches[pl]+'CodedArea', usecols = (4))
                probes[dictKey] = copy.deepcopy(probesOrig[dictKeyOrig])
                probes[dictKey]['coords'] = newCoords
                
                np.savetxt(str('probesToDat/'+dictKey+'.dat') ,np.transpose([probes[dictKey]['coords'][srtr,0], probes[dictKey]['coords'][srtr,1], probes[dictKey]['coords'][srtr,2], 
                            probes[dictKey]['meanCp'][srtr], probes[dictKey]['rmsCp'][srtr], probes[dictKey]['peakminCp'][srtr], probes[dictKey]['peakMaxCp'][srtr],tList[srtr],probes[dictKey]['Area'][srtr]]),
                            header = 'x'+' '*17+'y' +' '*17+'z'+' '*17+'meanCp'+' '*13+'rmsCp'+' '*13+'peakminCp'+' '*9+'peakMaxCp'+' '*9+'tapCode'+'    t0='+str(t0)+' tf='+str(tf),
                            fmt = '%1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %d    %1.12f ')
                
    return probes

def computeIntegralLoads(patches, angles, deltas, p = [0.5, 0, 0]):
            
    probes = {}
    
    for lvl in deltas:
        for ang in angles:
            
    
            cosine = np.cos(-ang*np.pi/180)
            sine   = np.sin(-ang*np.pi/180)

            rotMat = np.matrix([[cosine, 0, -sine],[0, 1, 0],[sine, 0, cosine]])

            for pl in patches:
                
                dictKey     = lvl + str(ang) + pl
                probes      = readDat([pl], [ang], [lvl], directory = 'probesToDat/')
                vertices    = np.loadtxt('probesLocation/'+patches[pl]+'Vertices', usecols = (5,6,7,8,9,10))
                normals     = np.loadtxt('probesLocation/'+patches[pl]+'Vertices', usecols = (11,12,13)).T
                srtr, tList = sortData(probes[dictKey]['coords'], patches[pl])
                
                xmin = vertices[:,0]
                xMax = vertices[:,1]
                ymin = vertices[:,2]
                yMax = vertices[:,3]
                zmin = vertices[:,4]
                zMax = vertices[:,5]
                
                bX = 0.5*(xmin+xMax) - p[0]
                bY = 0.5*(ymin+yMax) - p[1]
                bZ = 0.5*(zmin+zMax) - p[2]
                
                braccio = np.matrix([bX,bY,bZ]).T
                
                CfFaceAxes = -(np.multiply(np.multiply(probes[dictKey]['meanCp'],probes[dictKey]['Area']),normals)).T
                CfWindAxes = np.round((rotMat*(CfFaceAxes.T)).T, decimals = 12)
                                
                CmFaceAxes = np.cross(braccio,CfFaceAxes)/2.0                
                CmWindAxes = np.round((rotMat*(CmFaceAxes.T)).T, decimals = 15)
                
                
                print(pl + ' Force  ' + str(sum(CfWindAxes)))
                print(pl + ' Moment ' + str(sum(CmWindAxes)))
                
                np.savetxt(str('temp/'+dictKey+'.dat') ,np.transpose([probes[dictKey]['coords'][srtr,0], probes[dictKey]['coords'][srtr,1], probes[dictKey]['coords'][srtr,2], 
                            probes[dictKey]['meanCp'][srtr], probes[dictKey]['rmsCp'][srtr], probes[dictKey]['peakminCp'][srtr], probes[dictKey]['peakMaxCp'][srtr],tList[srtr],probes[dictKey]['Area'][srtr],
                            CfWindAxes[srtr,0],CfWindAxes[srtr,1],CfWindAxes[srtr,2],CmWindAxes[srtr,0],CmWindAxes[srtr,1],CmWindAxes[srtr,2]]),
                            header = 'x'+' '*17+'y' +' '*17+'z'+' '*17+'meanCp'+' '*13+'rmsCp'+' '*13+'peakminCp'+' '*9+'peakMaxCp'+' '*7+'tapCode'+' '*3+'Area'+' '*13+'Cf_x'+' '*11+'Cf_y'+' '*11+'Cf_z'+' '*11+'Cm_x'+' '*11+'Cm_y'+' '*11+'Cm_z',
                            fmt = '%1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %d    %1.12f %1.12f %1.12f %1.12f %1.12f %1.12f %1.12f')
                
    return probes
        
def readDat(patches, angles, deltas, directory = 'probesToDat/'):
    
    roundoff = 12
    probes = {}
    
    for lvl in deltas:
        for ang in angles:
            for pl in patches:
                
                if ang<0:
                    ang = 360+ang
                if ang >360:
                    ang = ang-360
                    
                dictKey = lvl + str(ang) + pl
                
                #fName = str('probesToDat/'+dictKey+'.dat')
                fName = str(directory+dictKey+'.dat')
                
                temp = np.loadtxt(fName, skiprows=1)
                
                coords = np.transpose(np.array([temp[:,0],temp[:,1],temp[:,2]]))
                
                probes[dictKey] = {'coords': coords, 'meanCp': np.around(temp[:,3],roundoff), 'rmsCp': np.around(temp[:,4],roundoff)
                                , 'peakminCp': np.around(temp[:,5],roundoff), 'peakMaxCp': np.around(temp[:,6],roundoff), 'Area': np.around(temp[:,8],roundoff),
                                'Cf_x' : temp[:,9],'Cf_y' : temp[:,10],'Cf_z' : temp[:,11],'Cm_x' : temp[:,12],'Cm_y' : temp[:,13],'Cm_z' : temp[:,14]}

    return probes

def readHRBDataset(deltas, angles, patches, t0, tf, nCpu, q, output):
    
    if output == 'file':
        input("Are you sure you want to read the probes from scratch?")
        input("Are you really sure?")
        
    probes = {}
    
    pool = mp.Pool(mp.cpu_count())
    for lvl in deltas:
        for ang in angles:
            for pl in patches:
                    
                ang = int(ang)
                case = lvl + str(ang)
                dictKey = lvl + str(ang) + pl
                print(dictKey)
            
                directory = str('../HRBDataset/' + lvl + '/' + str(ang) + '/probes' + case)

                coords   = rotateTranslateCoords(np.loadtxt(directory + '/' + patches[pl] + '.pxyz', skiprows=1, usecols = (1,2,3)),ang)
                srtr, tList = sortData(coords, patches[pl])
                
                pressure_list = Parallel(n_jobs=nCpu)(delayed(readPressure)(counter, directory, patches[pl])
                                for counter in range(t0,tf+1))
                
                pressure = np.array(pressure_list).transpose()
                
                peakminP = peakPressure(pressure, 0.22, 6.0, deltas[lvl],'min')
                peakMaxP = peakPressure(pressure, 0.22, 6.0, deltas[lvl],'Max')
                
                area = np.loadtxt('probesLocation/'+patches[pl]+'CodedArea', usecols = (4))
                
                probes[dictKey] = {'coords': coords, 'meanCp': np.mean(pressure,1)/q, 'rmsCp': np.sqrt(np.var(pressure,1))/q
                                 , 'peakminCp': peakminP/q, 'peakMaxCp': peakMaxP/q, 'Area': area}
                
                if output == 'file':
                    np.savetxt(str('probesToDat/'+dictKey+'.dat') ,np.transpose([probes[dictKey]['coords'][srtr,0], probes[dictKey]['coords'][srtr,1], probes[dictKey]['coords'][srtr,2], 
                            probes[dictKey]['meanCp'][srtr], probes[dictKey]['rmsCp'][srtr], probes[dictKey]['peakminCp'][srtr], probes[dictKey]['peakMaxCp'][srtr],tList[srtr],area]),
                            header = 'x'+' '*17+'y' +' '*17+'z'+' '*17+'meanCp'+' '*13+'rmsCp'+' '*13+'peakminCp'+' '*9+'peakMaxCp'+' '*9+'tapCode'+' Area'+'    t0='+str(t0)+' tf='+str(tf),
                            fmt = '%1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %1.12f    %d   %1.12f')
    pool.close()
    pool.join()
    if output == 'file':                
        return
    elif output == 'data':
        return probes
        
#########################################################################################

#uStar = 0.4962
#z0 = 0.0032
#Uref = uStar/0.41*np.log(1.975/z0)
#q = 0.5*1.225*Uref**2

#########################################################################################

#patches = {'F':'front','L':'leeward','R':'rear','T':'top','W':'windward'}
##patches = {'F':'front'}
#angles = list(range(0,360,10))
#deltas = {'Coarsest':0.0008,'Coarse':0.0005,'Fine':0.0003}
#probes = readDat(patches, angles, deltas)
#quantities = ['Cf_x','Cf_y','Cf_z','Cm_x','Cm_y','Cm_z']
##for qty in quantities:
    ##plotQty(probes, deltas, angles, patches, qty, [0,0], './Plots/IntegralLoads/')

#computeIntegralLoads(patches, angles, deltas)
##quantities = ['meanCp','rmsCp','peakminCp','peakMaxCp']
##scale = {'meanCp': [0,0],'rmsCp': [0, 0],'peakminCp': [0,0],'peakMaxCp': [0,0]}

##makeCasePeriodic(patches, list(range(0,360,10)), deltas, t0, tf)

##readHRBDataset(deltas, angles, patches, t0, tf, nCpu, q, 'file')
##probes = readDat(patches, angles, deltas)

##for qty in quantities:
    ##plotQty(probes, deltas, angles, patches, qty, scale[qty])


#########################################################################################
        
#uStar = 0.4962
#z0 = 0.0032
#Uref = uStar/0.41*np.log(1.975/z0)
#q = 0.5*1.225*Uref**2
#nCpu = 12
##t0 =   40001
##tf =  240000

#patches = {'F':'front','L':'leeward','R':'rear','T':'top','W':'windward'}
#tStep = 0.0003
#angles = [0,10,20,30,40,50,60,70,80,90]
#deltas = {'Fine':tStep}

#readHRBDataset(deltas, angles, patches, t0, tf, nCpu, q, 'file')

#for ang in angles:
    
    ####Coarsest Case
    ##t0 =   25000
    ##tf =  145000
    ##dt =    1250
    
    ####Coarse Case
    ##t0 =   40001
    ##tf =  240000
    ##dt =    1999
    
    ###Coarse Case
    #t0 =   66700
    #tf =  400001
    #dt =    3333
    #print(angles)
    #lastStep = np.arange(t0+dt,tf,dt)

    #nTaps = 0
    #nLags = len(lastStep)
    #truth = readDat(patches, [ang], deltas, directory = 'probesToDat/')

    #for lvl in deltas:
        #for pl in patches:
            
            #dictKey = lvl + str(ang) + pl
            #nTaps += len(truth[dictKey]['meanCp'])

    #meanCp = np.zeros((nTaps, len(lastStep)))
    #rmsCp = np.zeros(meanCp.shape)
    #peakMaxCp = np.zeros(meanCp.shape)
    #peakminCp = np.zeros(meanCp.shape)

    #print(lastStep)

    #cont = 0
    #for tf in lastStep:
        
        #print(tf)
        #probes = readHRBDataset(deltas, [ang], patches, t0, tf, nCpu, q, 'data')
        
        #idx = 0
        #nTaps = 0
        
        #for lvl in deltas:
            #for pl in patches:
                ##print(pl)
                #dictKey = lvl + str(ang) + pl
                #nTaps += len(probes[dictKey]['meanCp'])
                
                #meanCp[idx:nTaps,cont]    = probes[dictKey]['meanCp']
                #rmsCp[idx:nTaps,cont]     = probes[dictKey]['rmsCp']
                #peakMaxCp[idx:nTaps,cont] = probes[dictKey]['peakMaxCp']
                #peakminCp[idx:nTaps,cont] = probes[dictKey]['peakminCp']
                
                #idx = nTaps
                
        #cont+=1

    #cont = 0
    #for tf in lastStep:
        ##meanCp[:,cont]    = 100*np.absolute((meanCp[:,cont]    - meanCp[:,nLags-1])   /meanCp[:,nLags-1])
        ##rmsCp[:,cont]     = 100*np.absolute((rmsCp[:,cont]     - rmsCp[:,nLags-1])    /rmsCp[:,nLags-1])
        ##peakMaxCp[:,cont] = 100*np.absolute((peakMaxCp[:,cont] - peakMaxCp[:,nLags-1])/peakMaxCp[:,nLags-1])
        ##peakminCp[:,cont] = 100*np.absolute((peakminCp[:,cont] - peakminCp[:,nLags-1])/peakminCp[:,nLags-1])
        #meanCp[:,cont]    = np.absolute(meanCp[:,cont]    - meanCp[:,nLags-1])   
        #rmsCp[:,cont]     = np.absolute(rmsCp[:,cont]     - rmsCp[:,nLags-1])    
        #peakMaxCp[:,cont] = np.absolute(peakMaxCp[:,cont] - peakMaxCp[:,nLags-1])
        #peakminCp[:,cont] = np.absolute(peakminCp[:,cont] - peakminCp[:,nLags-1])

        #cont+=1
            
    #x = np.transpose((lastStep-t0)*tStep)
    #y = np.transpose(np.arange(0,nTaps))
    #xPlot = np.tile(x,(nTaps,1))
    #yPlot = np.transpose(np.tile(y,(nLags,1)))

    #levels = np.arange(0,0.01,0.001)

    #print(xPlot.shape)
    #print(yPlot.shape)
    
    #plt.contourf(xPlot, yPlot, meanCp, levels = levels, cmap = 'viridis')
    #plt.colorbar()
    #plt.plot(np.squeeze(lastStep[-2:-1]-t0)*tStep*np.ones(y.shape),y,color='lime',linewidth = 2.0)
    #plt.plot(np.squeeze(lastStep[-3:-2]-t0)*tStep*np.ones(y.shape),y,color='lime',linewidth = 2.0)
    #titolo = lvl + str(ang) + ' meanCp'
    #plt.yticks(np.array([189,630,819,1008,1449])-1, ['Front','Lee','Rear','Top','Wind'])
    #plt.xlabel('Averaging time [s]')
    #plt.title(titolo)
    #plt.savefig('Plots/ConvergenceCheck/' + titolo)
    ##plt.show(block=True)
    #plt.close()
    
    #plt.contourf(xPlot, yPlot, rmsCp, levels = levels, cmap = 'viridis')
    #plt.colorbar()
    #plt.plot(np.squeeze(lastStep[-2:-1]-t0)*tStep*np.ones(y.shape),y,color='lime',linewidth = 2.0)
    #plt.plot(np.squeeze(lastStep[-3:-2]-t0)*tStep*np.ones(y.shape),y,color='lime',linewidth = 2.0)
    #titolo = lvl + str(ang) + ' rmsCp'
    #plt.yticks(np.array([189,630,819,1008,1449])-1, ['Front','Lee','Rear','Top','Wind'])
    #plt.xlabel('Averaging time [s]')
    #plt.title(titolo)
    #plt.savefig('Plots/ConvergenceCheck/' + titolo)
    ##plt.show(block=True)
    #plt.close()
    
    #plt.contourf(xPlot, yPlot, peakminCp, levels = levels, cmap = 'viridis')
    #plt.colorbar()
    #plt.plot(np.squeeze(lastStep[-2:-1]-t0)*tStep*np.ones(y.shape),y,color='lime',linewidth = 2.0)
    #plt.plot(np.squeeze(lastStep[-3:-2]-t0)*tStep*np.ones(y.shape),y,color='lime',linewidth = 2.0)
    #titolo = lvl + str(ang) + ' peakminCp'
    #plt.yticks(np.array([189,630,819,1008,1449])-1, ['Front','Lee','Rear','Top','Wind'])
    #plt.xlabel('Averaging time [s]')
    #plt.title(titolo)
    #plt.savefig('Plots/ConvergenceCheck/' + titolo)
    ##plt.show(block=True)
    #plt.close()
    
    #plt.contourf(xPlot, yPlot, peakMaxCp, levels = levels, cmap = 'viridis')
    #plt.colorbar()
    #plt.plot(np.squeeze(lastStep[-2:-1]-t0)*tStep*np.ones(y.shape),y,color='lime',linewidth = 2.0)
    #plt.plot(np.squeeze(lastStep[-3:-2]-t0)*tStep*np.ones(y.shape),y,color='lime',linewidth = 2.0)
    #titolo = lvl + str(ang) + ' peakMaxCp'
    #plt.yticks(np.array([189,630,819,1008,1449])-1, ['Front','Lee','Rear','Top','Wind'])
    #plt.xlabel('Averaging time [s]')
    #plt.title(titolo)
    #plt.savefig('Plots/ConvergenceCheck/' + titolo)
    ##plt.show(block=True)
    #plt.close()
    
            
            
            
        
        
        
        
        
            
