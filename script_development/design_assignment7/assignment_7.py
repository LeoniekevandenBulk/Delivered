
# coding: utf-8

# # Vessel segmentation in retina fundus images (revisited)

# In this assignment, we are going to develop a system to automatically **segment vessels** in human retina fundus images (again).
# 
# In assignment 2 we trained a classical classification algorithm based on a Gaussian filterbank and a kNN classifier.
# This time we are going to use convolutional neural networks to solve the same task!
# 
# We are going to use the same data from the publicly available DRIVE dataset (http://www.isi.uu.nl/Research/Databases/DRIVE/).
# The DRIVE dataset consists of 40 images, 20 used for training and 20 used for testing. Each case contains:
# * fundus (RGB) image
# * a binary mask, which indicates the area of the image that has to be analyzed (removing black background)
# * manual annotations of retina vessels, provided as a binary map.
# 
# You can download the data for this assignment from this link: https://surfdrive.surf.nl/files/index.php/s/VZnaAZ8GWTZCCka (password: ismi2017)

# ## Tasks for this assignment

# 1. Develop a system to segment vessels in retina image in this notebook. You will have to submit this notebook with your code, which we will run and evaluate, together with the results of the segmentation.
# 2. Use the training set provided with the DRIVE dataset to train/tune the parameters of your system. You cannot use data from the test set available on the DRIVE website, nor from other datasets. 
# 3. Apply it to the test dataset and generate a binary map of the segmented vessel. The map must have the same size as the input image.
# 4. Submit the results of the notebook to the mini-challenge framework by running the corresponding cell in this notebook (at the end of the notebook). This will automatically submit your results to the mini-challenge framework. You will be able to visualize your scores in the **CHALLENGE** section of the webpage (http://ismi17.diagnijmegen.nl/). Note that after you submit the notebook, we will run your implementation and reproduce your result in order to evaluate your assignment. Any significant discrepancy between the results submitted to the mini-challenge framework and the one computed using this notebook will be penalized and discussed with the student.


# import libraries needed for this assignment
import os
import numpy as np
from PIL import Image
from tqdm import tnrange, tqdm_notebook
from scipy.ndimage.interpolation import affine_transform
from math import cos, sin, radians, floor, ceil

import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L

import random

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
get_ipython().magic(u'matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (20, 12)
from IPython import display
import time
from tqdm import tnrange, tqdm_notebook

from challenger import submit_results

print theano.config.floatX













# ## Shift and stitch
# You can immediately notice that an output is not produced for every pixel value.
# This is due to the downsampling that is performed in the network. 
# The network acts like a filter of size 'patch_size' (the size of the training patches) with a stride of 2^(number of max pooling operations present in the network). So with 2 max pooling operations in the network we obtain a stride of 4 pixels.
# 
# If we want to produce an output for every pixel, we can use a **shift-and-stitch** technique, i.e. we shift the input image by one pixel at a time and then stitch the results. More information can be found in this paper: P. Sermanet et al. Overfeat: Integrated recognition, localization and detection using convolutional networks. https://arxiv.org/abs/1312.6229
# 
# In the next function you will have to implement ```shift_and_stitch()```.

# In[182]:

def shift_and_stitch(im, patch_size, stride):
    ''' Return a full resolution segmentation by segmenting shifted versions
    of the input 'im' and stitching those segmentations back together. 
    The stride determines how many times the image should be shifted in the x and y direction'''
    
    patch_size = patch_size[0] 
    
    ## create the output shape, the output image is made a bit bigger to allow the stitched versions to be added. 
    ## the extra parts will be cut off at the end
    output = np.zeros([im.shape[0]+2*stride, im.shape[1]+2*stride]).astype(np.float32)
    
    # the input image has to be padded with zeros to allow shifting of the image. 
    # pad input image (half filter size + stride)
    im_padded = np.pad(im, ((patch_size//2, patch_size//2 + stride),
                            (patch_size//2, patch_size//2 + stride)), 'constant', constant_values = [0,0]).astype(np.float32)    
    im_p_sh = im_padded.shape
    
    # Now implement a loop that:
    # - obtains a shifted version of the image
    # - applies the fully convolutional network
    # - and places the network output in the output of this function
    
    im_g_padded = np.zeros((im_p_sh[0], im_p_sh[1]), dtype=np.float32)
        
    for row in range(0, stride):
        for col in range(0, stride):   
            
            for i in range(im_p_sh[0]-stride+1):
                for j in range(im_p_sh[1]-stride+1):
                        im_g_padded[i, j]=im_padded[i+row, j+col]
            # forward pass
            probability = evaluation_fn(np.expand_dims(np.expand_dims(im_g_padded.astype(np.float32), axis=0),axis=1))
            probability = probability[0,1,:,:]
            
            for i in range(probability.shape[0]):
                for j in range(probability.shape[1]):
                    output[i*stride+row,j*stride+col]=probability[i,j]
    
    return output[0:im.shape[0], 0:im.shape[1]] 


# ### Apply shift-and-stitch
# When you have succesfully implemented the shift and stitch method you can use it in the following function to segment the retinal vessels at full resolution and submit your results to the challenger framework

# In[199]:

# define these parameters
stride = 4
threshold = 0.85

f = 0
for img, msk in zip(tes_imgs,tes_msks):  
    
    img = np.asarray(Image.open(img))
    img_g = img[:,:,1].squeeze().astype(float)/255.0  
    msk = np.asarray(Image.open(msk))/255
    
    probability = shift_and_stitch(img_g.astype(np.float32), patch_size, stride)
    output = (probability > threshold) * msk
        
    plt.subplot(1,3,1)
    plt.imshow(img_g)
    plt.subplot(1,3,2)
    plt.imshow(probability)
    plt.subplot(1,3,3)
    plt.imshow(output)
    plt.show()
    
    result = Image.fromarray((255*output).astype('uint8'))
    result.save(os.path.join(result_output_folder, str(f+1) + "_mask.png")) 
    f += 1


# ## Submit your results to Challengr

# In[200]:

import challenger

challenger.submit_results({'username': 'G.Mooij',
                           'password': 'HEUJKXDW'},
                          result_output_folder,
                          {'notes': 'first'})


# # Additional data augmentation
# As an optional task, you can implement additional data augmentation by adding functions to the ```SampleExtractor``` class. You can retrain the proposed architecture and check if further data augmentation can improve results.
# Furthermore, you can modify the network architecture by adding/removing layers, adding dropout, L2 regularization, batch normalization, etc. **Be creative!**
# 
# 
# # Submission
# Document all your experiments, add new cells to this notebook if necessary.
# Submit the notebook fully executed to Freerk (freerk.venhuizen@radboudumc.nl) and Mehmet (mehmet.dalmis@radboudumc.nl). 

# In[ ]:



