# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:06:56 2022

@author: Preston
"""

#%%
# Loading Libraries

#Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras 

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


#%%  
  
# GPU Run this

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%%