
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import affine_transform
import random
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
matplotlib.rcParams['figure.figsize'] = (20, 12)

'''
Build a sample extractor

Differently from what done in the past two weeks, in this assignment we are going to implement a 'SampleExtractor'
class, which extract random samples from a list of images, in order to generate mini-batches on-the-fly. This means
that you don't need to create a dataset beforehand and then use it to train you network, but you will just have a list
of training images available, and patches will be extracted on-the-fly during training. This strategy allows to save
time in the preparation of your static dataset, and allows the use of a dynamic generation of mini-batches, where data
augmentation can also be applied on the fly. Note that this approach allows to test different strategies of data
augmentation without the need for making a new dataset from scratch all the time. We will implement two kind of data
augmentation in the class SampleExtractor:

    patch rotation
    patch flipping

In both cases, the event will occur at random, meaning that during patch extraction, some of the extracted patches will
be randomly transformed. Furthermore, the rotation angle will also be defined randomly, which means that the combination
of different patches used for training is in practice infinite (of course there is a limitation at some point).
'''

class SampleExtractor:
    def __init__(self, patch_size, vol, seg, max_rotation=0, rnd_flipping=False):
        self.patch_size = patch_size  # y,x

        # parameters used for data augmentation on-the-fly
        self.max_rotation = max_rotation  # float
        self.rnd_flipping = rnd_flipping  # boolean

        ## load images, masks, and labels in memory
        ## This is very slow, so do only once per image
        self.vol_array = nib.load(vol).get_data()
        self.seg_array = nib.load(seg).get_data()  # (0, 1)

        ## precalculate positive and negative indices inside the retina mask
        self.lbl_max = np.max(np.max(self.seg_array, axis=1), axis=0) # maximum label per slice
        self.lbl_max_0_idx = np.where(self.lbl_max == 0)[0]  # slice indices of slices with maximum label 0
        self.lbl_max_1_idx = np.where(self.lbl_max == 1)[0]  # slice indices of slices with maximum label 1
        self.lbl_max_2_idx = np.where(self.lbl_max == 2)[0]  # slice indices of slices with maximum label 2

        print('3D volume'+vol+', '+str(len(self.lbl_max))+' slices with max lbl=0, 1, 2: '+
              str(len(self.lbl_max_0_idx))+', '+str(len(self.lbl_max_1_idx))+', '+str(len(self.lbl_max_2_idx)))

        #def get_rnd_rotation(self):
    #    ''' Return a 2x2 rotation matrix to rotate by a random angle phi.
    #        The angle should in the range of -self.max_rotations to +self.max_rotations. '''
    #    if self.max_rotation == 0:
    #        return np.eye(2)  # identity matrix
    #    else:
    #        # create a random rotation matrix 'rot_mat' in the range of -'self.max_rotations' to +'self.max_rotations'

    #        phi = np.random.uniform(-1, 1) * self.max_rotation * np.pi / 180

    #        rot_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    #        return rot_mat  # 2x2 np.array

    #def rnd_flip(self, X):
    #    ''' Return a flipped version of the input patch X.
    #        The kind of flip and the probability that the event happens
    #        should be based on a random event. '''

        # write a function that randomly flips the input image

    #    if np.random.random() < 0.5:
    #        X = np.flipud(X)
    #    else:
    #        X = np.fliplr(X)

    #    return X

    def extract_sample(self, slice):
        #ps_y, ps_x = self.patch_size

        X = self.vol_array[:, :, slice]

        Y = self.seg_array[:, :, slice]

        ## get a random rotation matrix
        #rot_mat = self.get_rnd_rotation()

        ## Shift coordinate system to the center of the image
        ## for more information http://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
        #c_in = loc_y, loc_x
        #c_out = np.array([ps_y / 2, ps_x / 2])
        #offset = c_in - c_out.dot(rot_mat)

        ## extract the patch and apply the rotation matrix
        #X = affine_transform(self.vol_array[image], rot_mat.T, offset=offset, output_shape=(ps_y, ps_x),
        #                     order=1) / 255.0
        ## extract the label
        #Y = int(self.lbl_array[image, loc_y, loc_x] > 0)

        X = X.astype(np.float32)

        Y = Y.astype(np.int16)

        ## apply random flipping
        #if self.rnd_flipping:
        #    X = self.rnd_flip(X)

        return X, Y # 512x512 slice of each of the 3D arrays

    def get_random_sample_from_class(self, label):
        ''' Extract a patch with a given label. '''

        # tip: use 'self.pos_idx' and 'self.neg_idx'
        if label == 0:
            slice = self.lbl_max_0_idx[int(random.random() * len(self.lbl_max_0_idx))]
        elif label == 1:
            slice = self.lbl_max_1_idx[int(random.random() * len(self.lbl_max_1_idx))]
        elif label == 2:
            slice = self.lbl_max_2_idx[int(random.random() * len(self.lbl_max_2_idx))]
        else:
            print("Unknown label"+str(label))

        return self.extract_sample(slice)

#'''
#A convenience function to visualize the output of the patch extractor
#'''

#def visualize_samples(sampling_function, label):
#    for i in range(5):
#        X, Y = sampling_function(label)
#        plt.subplot(1,5,i+1)
#        plt.imshow(X, cmap='bone', vmin=0, vmax=1); plt.title('patch: ' + str(i) + " , class: " + str(Y))