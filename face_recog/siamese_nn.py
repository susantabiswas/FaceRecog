# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Driver code to load facenet model which is a 
siamese neural network.'''
# ===================================================

from keras.models import load_model
import tensorflow as tf 
import os 
import numpy as np 
import cv2 

class FaceNet:
    MODEL_DIM = 96
    def __init__(self, model_loc:str='models') -> None:
        self.model_loc = os.path.join(model_loc,
                                     'facenet.h5')
        # Load facenet
        self.facenet = self.load_facenet()


    def triplet_loss(self, y_true, y_pred, alpha = 0.2):
        # The model makes an encoding vector consisting of 128 numbers for the input image. 
        # Two encodings are compared and if the two encodings are similar then we say that 
        # the two images are of the same person otherwise they are different. 
        # The model uses Triplet loss function. The aim is to minimize this function.
        # Triplet loss function
        #  y_pred - list containing three objects:
        #         anchor(None, 128) -- encodings for the anchor images
        #         positive(None, 128) -- encodings for the positive images
        #         negative(None, 128) -- encodings for the negative images

        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        
        # triplet formula components
        pos_dist = tf.reduce_sum( tf.square(tf.subtract(y_pred[0], y_pred[1])) )
        neg_dist = tf.reduce_sum( tf.square(tf.subtract(y_pred[0], y_pred[2])) )
        basic_loss = pos_dist - neg_dist + alpha
        
        loss = tf.maximum(basic_loss, 0.0)
    
        return loss


    def load_facenet(self):
        # The model outputs a vector of 128 numbers which represent encoding for the
        # given input image. We will be using this encoding vector for comparing two images.
        # This network takes as input 96x96 RGB image as its input. 
        # Specifically, inputs a tensor of shape $(m, n_C, n_H, n_W)$ , where $n_C$ = channel.
        # Output
        # - A matrix of shape **(m, 128)** where the 128 numbers are the encoding values for $ith$ image.
        
        self.facenet = load_model(self.model_loc, 
                        custom_objects={
                            'triplet_loss': self.triplet_loss
                        })
        print(self.facenet)


    def get_image_encodings(self, image):
        # resize the image to 96 x 96
        img_cropped = cv2.resize(image, 
                                (FaceNet.MODEL_DIM,
                                 FaceNet.MODEL_DIM))
        
        img_cropped = img_cropped[...,::-1]
        img_cropped = np.around(np.transpose(
                                    img_cropped, 
                                    (2,0,1))/255.0,
                                    decimals=12)
        x_train = np.array([img_cropped])
        # Run a single forward propagation to get the outlayer encodings
        embedding = self.facenet \
                        .predict_on_batch(x_train)

        return embedding

if __name__ == "__main__":
    ob = FaceNet(model_loc='models')
    img = cv2.imread('data/sample/1.jpg')
    print(ob.facenet.summary())
    encodings = ob.get_image_encodings(img)
    print(encodings)

        