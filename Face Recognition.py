
# coding: utf-8

# # <u>Face Recognition System
# ### NOTE: This notebook doesn't contain all the features , for using those run the face_recognition.py script
# Nowadays there are many ways of authenticating yourself, like using password, retina scan, fingerprint etc. Face can also be used for this purpose. In this notebook we will make a face recognition system using Siamese network.
# This is different from face verification where the task is to know whether given two input images are same or not.
# Here the task is see whether the given input image is of any person who is registered with the system or not. There can be multiple users registered with the system.
# 
# The advantage of **Siamese Network** is that it allows a way to do this sort of verification task with very little user data, as it is quite unreasonable to train using thousands of images for each user. Here we will be using **FaceNet Model**.
# 
# FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. So by comparing two such vectors, we can then determine if two pictures are of the same person.

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utility import *
from inception_blocks_v2 import *
from webcam_utils import *
get_ipython().magic('matplotlib inline')
np.set_printoptions(threshold=np.nan)


# ## Model
# The model makes an encoding vector consisting of 128 numbers for the input image. Two encodings are compared and if the two encodings are similar then we say that the two images are of the same person otherwise they are different. 
# The model uses **Triplet loss function**. The aim is to minimize this function.

# In[2]:


# triplet loss function
#  y_pred - list containing three objects:
#         anchor(None, 128) -- encodings for the anchor images
#         positive(None, 128) -- encodings for the positive images
#         negative(None, 128) -- encodings for the negative images
def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # triplet formula components
    pos_dist = tf.reduce_sum( tf.square(tf.subtract(y_pred[0], y_pred[1])) )
    neg_dist = tf.reduce_sum( tf.square(tf.subtract(y_pred[0], y_pred[2])) )
    basic_loss = pos_dist - neg_dist + alpha
    
    loss = tf.maximum(basic_loss, 0.0)
   
    return loss


# ### Loading the Model
# The model outputs a vector of 128 numbers which represent encoding for the given input image. We will be using this encoding vector for comparing two images.
# #### Input
# - This network takes as input 96x96 RGB image as its input. Specifically, inputs a tensor of shape $(m, n_C, n_H, n_W)$ , where $n_C$ = channel.
# 
# #### Output
# - A matrix of shape **(m, 128)** where the 128 numbers are the encoding values for $ith$ image.

# In[3]:


# load the model
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])


# ### Loading the trained weights
# We will be using a pretrained model since it requires a lot of time and data for training such a model.

# In[4]:


# load saved weights
load_weights_from_FaceNet(FRmodel)


# ### User Database
# 
# We will create a databse of registered. For this we will use a simple dictionary and map each registered user with his/her face encoding.

# ### Add user Here

# In[5]:


# we use a dict for keeping track of ampping of each person with his/her face encoding
user_db = {}
# add a user
user_db["susanta"] = img_to_encoding("images/2.jpg", FRmodel)
user_db["person 2"] = img_to_encoding("images/4.jpg", FRmodel)


# ### Putting everything together
# For making this face recognition system we are going to take the input image, find its encoding and then see if there is any similar encoding in the database or not. We define a threshold value to decide whether the two images are similar or not based on the similarity of their encodings.

# In[15]:


# recognize the user face by checking for it in the database
def recognize_face(image_path, database, model):
    # find the face encodings for the input image
    encoding = img_to_encoding(image_path, model)
    
    min_dist = 99999
    threshold = 0.7
    # loop over all the recorded encodings in database 
    for name in database:
        # find the similarity between the input encodings and claimed person's encodings using L2 norm
        dist = np.linalg.norm(np.subtract(database[name], encoding) )
        # check if minimum distance or not
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > threshold:
        print("User not in the database.")
        identity = 'Unknown Person'
    else:
        print ("Hi! " + str(identity) + ", L2 distance: " + str(min_dist))
        
    return min_dist, identity


# ### <u>Run the face recognizer

# In[14]:


# we can use the webcam to capture the user image then get it recognized
detect_face()
img = resize_img("saved_image/1.jpg")
recognize_face("saved_image/1.jpg", user_db, FRmodel)


# ### References:
# - A lot of code is based on the assignment from Convolutional Neural Networks Specialization by Deeplearning.ai on Coursera.<br>
# https://www.coursera.org/learn/convolutional-neural-networks/home/welcome 
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
# - The pretrained model used is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - A lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
