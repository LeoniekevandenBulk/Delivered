# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:17:09 2017

@author: Wouter Eijlander
"""

import numpy as np
from PIL import Image
import scipy.ndimage.interpolation as sci
import random
import matplotlib
import matplotlib.pyplot as plt


class BatchAugmenter:
    def __init__(self, img_batch, img_labels, augment_params, max_rotation=10, gauss_avg = 0, gauss_std=10, max_deform = 0):
        self.max_rotation = max_rotation
        self.gauss_avg = gauss_avg
        self.gauss_std = gauss_std
        self.max_deform = max_deform
        self.img_batch = img_batch
        self.img_labels = img_labels
        # Note that the current algorithm assumes class 1 to be those containing lesions, and class 2 to contain none.
        self.class1_params = augment_params[0]
        self.class2_params = augment_params[1]
        
    def get_rnd_rotation(self, X, Y):
        ''' Return a rotated image and its labels. Both are rotated the same amount.
        The angle should in the range of -self.max_rotations to +self.max_rotations. '''
        dim1, dim2 = np.shape(X)
        dim1 = int(np.floor(dim1/2))
        dim2 = int(np.floor(dim2/2))
        rotation = np.random.uniform(-1,1)
        X = sci.rotate(X, rotation*self.max_rotation)
        Y = sci.rotate(Y, rotation*self.max_rotation)
        newposx1 = int(np.floor(np.shape(X)[0]/2)-dim1)
        newposx2 = int(np.floor(np.shape(X)[0]/2)+dim1)
        newposy1 = int(np.floor(np.shape(X)[1]/2)-dim2)
        newposy2 = int(np.floor(np.shape(X)[1]/2)+dim2)
        X = X[newposx1:newposx2, newposy1:newposy2].astype(int)
        Y = Y[newposx1:newposx2, newposy1:newposy2].astype(int)
        return X, Y
    
    def get_rnd_elastic(self):
        ''' Return a 2x2 deformation matrix to rotate by a random angle phi.
        The angle should in the range of -self.max_rotations to +self.max_rotations. '''
        return 0
        
        
    def get_gauss_noise(self, X):
        '''Return a single image and label with the same noise.'''
        noise = np.random.normal(self.gauss_avg, self.gauss_std, size=X.shape)
        X = np.clip(X+noise,0,255)
        return X
        
    def getAugmentation(self):
        augmented_img_batch = np.zeros(np.shape(img_batch[0]))
        augmented_img_labels = np.zeros(np.shape(img_labels[0]))
        for i in range(len(self.img_batch)):
            X = self.img_batch[i]
            Y = self.img_labels[i]
            
            # Determine what class we're dealing with, select parameters accordingly, and decide on the fly which to perform
            if np.any(self.img_labels[i]) == 2:
                rotation, elastic, gauss = [random.uniform(0,1) > x for x in self.class1_params]
            else:
                rotation, elastic, gauss = [random.uniform(0,1) > x for x in self.class2_params]
            
            # Perform augmentations
            if rotation:
                X, Y = self.get_rnd_rotation(X,Y)
            if elastic:
                '''doesn't do anything currently- add elastic deform if time permits.'''
            if gauss:
                # Not applid to the labels; we're not actually creating lesions, just adding nise to the input
                X = self.get_gauss_noise(X)

            augmented_img_batch = np.dstack((augmented_img_batch,X))
            augmented_img_labels = np.dstack((augmented_img_labels,Y))
        return augmented_img_batch[:,:,1:], augmented_img_labels[:,:,1:]
        
img1 = Image.open("assignment7/test2.jpeg")
img2 = Image.open("assignment7/test2.jpeg")
img_batch = [np.array(img1)[:,:,1], np.array(img2)[:,:,1]]
img_labels = [np.array(img1)[:,:,1], np.array(img2)[:,:,1]]
myAugmenter = BatchAugmenter(img_batch, img_labels, [[0.1,0.8,0.9],[0.1,0.8,0.6]])
img_batch, img_labels = myAugmenter.getAugmentation()
print np.shape(img_batch)
print np.shape(img_labels)
#Image.fromarray(img_batch[:,:,0]).show()
#Image.fromarray(img_labels[:,:,0]).show()
