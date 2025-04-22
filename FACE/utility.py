#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

def torch_to_numpy(tensor):
    try:
        return tensor.detach().cpu().numpy()
    except:
        return np.array(tensor)


# In[ ]:




