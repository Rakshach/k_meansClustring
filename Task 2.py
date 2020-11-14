#!/usr/bin/env python
# coding: utf-8

# ### By Raksha Choudhary 
# ### Data Science and Bussiness Analytics Internship
# ### GRIP The Sparks Foundation
# ### Task 2 : From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.

# ###   Import libraries 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans


# ### Load Iris Dataset

# #### The iris Dataset consist of 3 types of irises(Setosa,Versicolour and Virginica) petal and sepal length stored in a 150*4 numpy

# In[2]:


iris = datasets.load_iris()
iris_df =pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head()   #giving first five values


# ### Finding optimum number of cluster 
# #### Here we using k means algorithm for k_ means classification for Clustering. It is the process of dividing the entire data into groups (also known as clusters) based on the patterns in the data.In clustering, we do not have a target to predict. We look at the data and then try to club similar observations and form different groups.

# In[3]:


# Finding the optimum number of clusters for k-means classification

x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    


# ### Ploting the graph

# In[7]:



# Plotting the results onto a line graph 

plt.plot(range(1, 11), wcss,c='r')
plt.style.use("dark_background")
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')# Within cluster sum of squares
plt.tight_layout()
plt.show()


# #### the number of cluster we found is 3

# #### From the above graph,we can see why it is called "The elbow method" the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration

# ### Creating k-means classifier

# In[5]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# ### Visualisation

# In[6]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
plt.tight_layout()

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# #### from above graph we see the difference between actual vs predicted
# We looked at the challenges which we might face while working with K-Means and also saw how K-Means++ can be helpful when initializing the cluster centroids.
# 
# Finally, we implemented k-means and looked at the elbow curve which helps to find the optimum number of clusters in the K-Means algorithm.
