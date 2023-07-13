#!/usr/bin/env python
# coding: utf-8

# In[3]:


import clip
import torch
from PIL import Image
import torch.nn.functional as tnf


# In[4]:


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# In[5]:


@torch.no_grad()
def get_ViT_feat(path):
    """
    path: the image path
    """
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    feature = model.encode_image(image)
    return feature


# In[ ]:




