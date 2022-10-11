import numpy as np
import pandas as pd
import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

labels = ['rmsCp']
variables = ['CfMean','TKE','U','gradP','meanCp']

dataFrame_00deg = pd.read_csv('Features/Coarsest0')
dataFrame_10deg = pd.read_csv('Features/Coarsest10')
dataFrame_30deg = pd.read_csv('Features/Coarsest30')
dataFrame_40deg = pd.read_csv('Features/Coarsest40')
dataFrame_90deg = pd.read_csv('Features/Coarsest90')

features_00deg = dataFrame_00deg[variables].values
features_10deg = dataFrame_10deg[variables].values
features_30deg = dataFrame_30deg[variables].values
features_40deg = dataFrame_40deg[variables].values
features_90deg = dataFrame_90deg[variables].values

labels_00deg = dataFrame_00deg[labels].values
labels_10deg = dataFrame_10deg[labels].values
labels_30deg = dataFrame_30deg[labels].values
labels_40deg = dataFrame_40deg[labels].values
labels_90deg = dataFrame_90deg[labels].values

# PCA
X_00deg = PCA(n_components=2).fit_transform(features_00deg)
X_10deg = PCA(n_components=2).fit_transform(features_10deg)
X_30deg = PCA(n_components=2).fit_transform(features_30deg)
X_40deg = PCA(n_components=2).fit_transform(features_40deg)    
X_90deg = PCA(n_components=2).fit_transform(features_90deg)

# TSNE
#X_00deg = TSNE(n_components=2).fit_transform(features_00deg)
#X_10deg = TSNE(n_components=2).fit_transform(features_10deg)
#X_90deg = TSNE(n_components=2).fit_transform(features_90deg)

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(5,5))
plt.plot(X_00deg[:,0], X_00deg[:,1], '.r', markersize=12)
plt.plot(X_10deg[:,0], X_10deg[:,1], '.g', markersize=12)
plt.plot(X_90deg[:,0], X_90deg[:,1], '.b', markersize=12)
plt.legend([r"$0^\circ$", r"$10^\circ$", r"$90^\circ$"])
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
#plt.savefig("TSNE_2comp_all_30+40.png")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_00deg[0::10,0], X_00deg[0::10,1], labels_00deg[0::10], c='r', marker='.')
ax.scatter(X_10deg[0::10,0], X_10deg[0::10,1], labels_10deg[0::10], c='g', marker='.')
ax.scatter(X_90deg[0::10,0], X_90deg[0::10,1], labels_90deg[0::10], c='b', marker='.')

ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
ax.set_zlabel(r"$C_p'$")

plt.legend([r"$0^\circ$", r"$10^\circ$", r"$90^\circ$"])
plt.show()
