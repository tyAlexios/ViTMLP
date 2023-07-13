#!/usr/bin/env python
# coding: utf-8

# In[152]:


import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[153]:


"""
data: Tensor | N x feat
feat == 1 x M
"""

def get_param(data, num_pos):
    
    num_samples = data.shape[0]
    size_feat = data.shape[1]
    
    features = np.array(data.cpu().detach())
    labels = torch.ones(num_pos)
    for i in range(2, num_samples+1):
        labels = torch.cat((labels, torch.ones(num_pos)*i), dim=0)
    
    labels = np.array(labels).astype(int)
    
    return features, labels, num_samples, size_feat


# In[154]:


def plot_embedding_2dim(inputs, labels ,title):
    x_min, x_max = np.min(inputs, 0), np.max(inputs, 0)
    inputs = (inputs-x_min)/(x_max-x_min) #normalization
    
    fig = plt.figure()
    ax = plt.subplot(111)
    
    for i in range(inputs.shape[0]):
        plt.text(inputs[i][0], 
                 inputs[i][1],
                 str(labels[i]),
                 color = plt.cm.Set1(labels[i]/10),
                 fontdict={'weight': 'bold', 'size': 7}
                )
    
    plt.xticks()
    plt.yticks()
    plt.title(title, fontsize=14)
    
    return fig


# In[155]:


def show2dim(show_data, num_pos):
    
    features, labels, num_samples, size_feat = get_param(show_data, num_pos)
    data_2dim = TSNE(n_components=2, init='pca', random_state=0).fit_transform(features)
    
    fig_2dim = plot_embedding_2dim(data_2dim, labels, 't-SNE Embedding')
    
    plt.show()



def show3dim(show_data, num_pos):
    
    features, labels, num_samples, size_feat = get_param(show_data, num_pos)
    data_3dim = TSNE(n_components=3).fit_transform(features)
    
    x_min, x_max = np.min(data_3dim, 0), np.max(data_3dim, 0)
    data_3dim = data_3dim/(x_max-x_min)
    
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    
    for i in range(data_3dim.shape[0]):
        ax.scatter(data_3dim[i, 0],
                   data_3dim[i, 1],
                   data_3dim[i, 2],
                   c=plt.cm.Set1(labels[i]/10)
                   )
    
    plt.axis('off')
    plt.show()



