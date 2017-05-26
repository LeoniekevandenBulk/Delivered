# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:17:09 2017

@author: Wouter Eijlander
"""

import numpy as np
from PIL import Image
import scipy.ndimage.interpolation as sci
import random
import cv2


class BatchAugmenter:
    def __init__(self, img_batch, img_labels, augment_params, group_labels="liver", max_rotation=10, gauss_avg = 0, gauss_std=10, max_deform = 0):
        self.max_rotation = max_rotation
        self.gauss_avg = gauss_avg
        self.gauss_std = gauss_std
        self.max_deform = max_deform
        self.img_batch = img_batch
        self.img_labels = img_labels
        # Note that the current algorithm assumes group 0 = lesion, group 1 = liver.
        self.class_params = augment_params[int(group_labels=="liver")]
        
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
    
    def get_rnd_elastic(self, X, Y):
        """
        Based on: https://www.kaggle.com/bguberfain/ultrasound-nerve-segmentation/elastic-transform-for-data-augmentation
        Transform an image elastically as a form of data augmentation
        # Params
        - image : the image to transform
    
        # Returns
        - the transformed image
        """
            
        ran = np.random.randint(4)
        alpha = X.shape[1] * ran
        sigma = X.shape[1] * 0.2
        alpha_affine = X.shape[1] * 0.035
        random_state = np.random.RandomState(None)
    
        shape = X.shape
        shape_size = shape[:2]
        
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        X = cv2.warpAffine(X, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        
        blur_size = int(4*sigma) | 1
        
        #Blur at half resolution
        dx = cv2.GaussianBlur(X[::4,::4], ksize=(blur_size, blur_size), sigmaX=sigma)
        dy = cv2.GaussianBlur(X[::4,::4], ksize=(blur_size, blur_size), sigmaX=sigma)
        
        dx = cv2.resize(dx, dsize=(dx.shape[0]*4, dx.shape[1]*4)).transpose(1,0,2)
        dy = cv2.resize(dy, dsize=(dy.shape[0]*4, dy.shape[1]*4)).transpose(1,0,2)
    
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    
        return sci.map_coordinates(X, indices, order=1, mode='reflect').reshape(shape), sci.map_coordinates(Y, indices, order=1, mode='reflect').reshape(shape)

        
        
    def get_gauss_noise(self, X):
        '''Return a single image and label with the same noise.'''
        noise = np.random.normal(self.gauss_avg, self.gauss_std, size=X.shape)
        return X+noise
        
    def getAugmentation(self):
        augmented_img_batch = np.zeros(np.shape(img_batch[0]))
        augmented_img_labels = np.zeros(np.shape(img_labels[0]))
        for i in range(len(self.img_batch)):
            X = self.img_batch[i]
            Y = self.img_labels[i]
            
            # Determine what class we're dealing with, select parameters accordingly, and decide on the fly which to perform
            rotation, elastic, gauss = [random.uniform(0,1) > x for x in self.class_params]
            
            # Perform augmentations
            if rotation:
                X, Y = self.get_rnd_rotation(X,Y)
            if elastic:
                X,Y = sellf.get_rnd_elastic(X,Y)
            if gauss:
                # Not applid to the labels; we're not actually creating lesions, just adding nise to the input
                X = self.get_gauss_noise(X)

            augmented_img_batch = np.dstack((augmented_img_batch,X))
            augmented_img_labels = np.dstack((augmented_img_labels,Y))
        return augmented_img_batch[:,:,1:], augmented_img_labels[:,:,1:]
