#!/usr/bin/env python
# coding: utf-8

# In[121]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np


# In[122]:


#Call required libraries
import time                   # To time processes
import warnings               # To suppress warnings

import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio 
import matplotlib.pyplot as plt                   # For graphics
import seaborn as sns


from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering
from sklearn.mixture import GaussianMixture #For GMM clustering

import os                     # For os related operations
import sys                    # For data size


# In[123]:


wh = pd.read_csv(r'C:\PythonTest\Capstonedatautf8nobadv3.csv')
wh.describe()


# In[157]:


print("Dimension of dataset: wh.shape")
wh.dtypes


# In[158]:


wh1 = wh[['business.leisure', 'days.booked.before.departure', 'frequent.flyer.airline', 'ff.flag', 
           'service.class.code', 'birth_decade',  
          'trip.start.port.code', 'fare.set', 'group.band.num', 
          'trip.start.port.state.num', 'trip.start.port.region.num',  
          'pp.country.code']] #Subsetting the data
cor = wh1.corr() #Calculate the correlation of the above variables
sns.heatmap(cor, square = False, robust = True) #Plot the correlation as heat map


# In[156]:


wh1 = wh[['business.leisure', 'days.booked.before.departure', 'frequent.flyer.airline', 'ff.flag', 
           'service.class.code', 'birth_decade',  
          'trip.start.port.code', 'fare.set', 'group.band.num', 
          'trip.start.port.state.num', 'trip.start.port.region.num',  
          'pp.country.code', 'age.generation.label']] #Subsetting the data
cor = wh1.corr() #Calculate the correlation of the above variables
sns.heatmap(cor, square = False, robust = True) #Plot the correlation as heat map


# In[159]:


wh1 = wh[['business.leisure', 'days.booked.before.departure', 
           'service.class.code', 'birth_decade',  
          'trip.start.port.code', 'fare.set',
          'trip.start.port.region.num'
         ]] #Subsetting the data
cor = wh1.corr() #Calculate the correlation of the above variables
sns.heatmap(cor, square = False, robust = True) #Plot the correlation as heat map


# In[161]:


wh1.dropna()


# In[162]:


wh1.head()


# In[163]:


np.any(np.isnan(wh1))


# In[165]:


np.all(np.isfinite(wh1))


# In[166]:


#Scaling of data
ss = StandardScaler()
ss.fit_transform(wh1)


# In[167]:


print(ss.fit(wh1))


# In[168]:


wh1.head()


# In[170]:


wh1.describe()


# In[171]:


#K means Clustering 
def doKmeans(X, nclust=2):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(wh1, 2)
kmeans = pd.DataFrame(clust_labels)
wh1.insert((wh1.shape[1]),'kmeans',kmeans)


# In[ ]:




