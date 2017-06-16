import numpy as np
from PIL import Image
import scipy.ndimage.interpolation as sci
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

'''
Batch Augmenter
'''

class BatchAugmenter:

    # Return a rotated image and its labels. Both are rotated the same amount
    # The angle should in the range of -max_rotations to +max_rotations
    def get_rnd_rotation(self, X, Y, M, max_rotation):
        dim1, dim2 = np.shape(X)
        dim1 = int(np.floor(dim1/2))
        dim2 = int(np.floor(dim2/2))
        rotation = np.random.uniform(-1,1)
        X = sci.rotate(X, rotation*max_rotation)
        Y = sci.rotate(Y, rotation*max_rotation)
        M = sci.rotate(M, rotation*max_rotation)
        newposx1 = int(np.floor(np.shape(X)[0]/2)-dim1)
        newposx2 = int(np.floor(np.shape(X)[0]/2)+dim1)
        newposy1 = int(np.floor(np.shape(X)[1]/2)-dim2)
        newposy2 = int(np.floor(np.shape(X)[1]/2)+dim2)
        X = X[newposx1:newposx2, newposy1:newposy2]
        Y = Y[newposx1:newposx2, newposy1:newposy2]
        M = M[newposx1:newposx2, newposy1:newposy2]
        return X, Y, M

    # Elastic deformation of images as described in [Simard2003]
    # Simard, Steinkraus and Platt, "Best Practices for Convolutional
    # Neural Networks applied to Visual Document Analysis", in Proc.
    # of the International Conference on Document Analysis and Recognition, 2003.
    def get_rnd_elastic(self, X, Y, alpha=35, sigma=6, random_state=None):
        """
        """
        assert len(X.shape)==2

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = X.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        
        return map_coordinates(X, indices, order=1).reshape(shape),map_coordinates(Y, indices, order=1).reshape(shape)

    # Apply gaussian noise on the image
    def get_gauss_noise(self, X, gauss_avg, gauss_std):
        '''Return a single image and label with the same noise.'''
        noise = np.random.normal(gauss_avg, gauss_std, size=X.shape)
        return X+noise

    # Main function to apply all augmentations to a batch depending on augment_params    
    def getAugmentation(self, img_batch, img_labels, img_masks, augment_params, max_rotation=10, gauss_avg = 0, gauss_percent=0.05):
        augmented_img_batch = np.zeros(np.shape(img_batch))
        augmented_img_labels = np.zeros(np.shape(img_labels))
        augmented_img_masks = np.zeros(np.shape(img_masks))
	gauss_std=(gauss_percent*(np.max(img_batch)-np.min(img_batch)))
        
        for i in range(np.shape(img_batch)[0]):
            X = np.squeeze(img_batch[i,:,:,:])
            Y = np.squeeze(img_labels[i,:,:,:])
            M = np.squeeze(img_masks[i,:,:,:])
            
            # Determine what class we're dealing with, select parameters accordingly, and decide on the fly which to perform
            rotation, elastic, gauss = [random.uniform(0,1) < x for x in augment_params]
            
            # Perform augmentations
            if rotation:
                X,Y,M = self.get_rnd_rotation(X,Y,M,max_rotation)
                Y = np.round(Y)
                M = np.round(M)
            if elastic:
                X,Y = self.get_rnd_elastic(X,Y)

            if gauss:
                # Not applied to the labels; we're not actually creating lesions, just adding noise to the input
                X = self.get_gauss_noise(X, gauss_avg, gauss_std)

            augmented_img_batch[i,:,:,:] = X
            augmented_img_labels[i,:,:,:] = Y
            augmented_img_masks[i,:,:,:] = M
            
        return augmented_img_batch, augmented_img_labels, augmented_img_masks
