
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import affine_transform
import random
import matplotlib
import matplotlib.pyplot as plt
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
    def __init__(self, patch_size, imgs, msks, lbls, max_rotation=0, rnd_flipping=False):
        self.patch_size = patch_size  # y,x
        self.img_array = np.zeros([len(imgs), 584, 565])
        self.msk_array = np.zeros([len(msks), 584, 565])
        self.lbl_array = np.zeros([len(lbls), 584, 565])

        # parameters used for data augmentation on-the-fly
        self.max_rotation = max_rotation  # float
        self.rnd_flipping = rnd_flipping  # boolean

        ## load images, masks, and labels in memory
        for i, (img, msk, lbl) in enumerate(zip(imgs, msks, lbls)):
            self.img_array[i] = np.asarray(Image.open(img))[:, :, 1]  # green channel
            self.msk_array[i] = np.asarray(Image.open(msk)) / 255.0  # (0, 1)
            self.lbl_array[i] = np.asarray(Image.open(lbl))

        ## precalculate positive and negative indices inside the retina mask
        self.pos_idx = np.array(
            np.where(np.logical_and(self.lbl_array > 0, self.msk_array == 1)))  # indexed as [image, y, x]
        self.neg_idx = np.array(
            np.where(np.logical_and(self.lbl_array == 0, self.msk_array == 1)))  # indexed as [image, y, x]

        print('shape of positive indexes: {}'.format(self.pos_idx.shape))
        print('shape of negative indexes: {}'.format(self.neg_idx.shape))

    def get_rnd_rotation(self):
        ''' Return a 2x2 rotation matrix to rotate by a random angle phi.
            The angle should in the range of -self.max_rotations to +self.max_rotations. '''
        if self.max_rotation == 0:
            return np.eye(2)  # identity matrix
        else:
            # create a random rotation matrix 'rot_mat' in the range of -'self.max_rotations' to +'self.max_rotations'

            phi = np.random.uniform(-1, 1) * self.max_rotation * np.pi / 180

            rot_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

            return rot_mat  # 2x2 np.array

    def rnd_flip(self, X):
        ''' Return a flipped version of the input patch X.
            The kind of flip and the probability that the event happens
            should be based on a random event. '''

        # write a function that randomly flips the input image

        if np.random.random() < 0.5:
            X = np.flipud(X)
        else:
            X = np.fliplr(X)

        return X

    def extract_sample(self, image, loc_y, loc_x):
        ps_y, ps_x = self.patch_size

        ## get a random rotation matrix
        rot_mat = self.get_rnd_rotation()

        ## Shift coordinate system to the center of the image
        ## for more information http://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
        c_in = loc_y, loc_x
        c_out = np.array([ps_y / 2, ps_x / 2])
        offset = c_in - c_out.dot(rot_mat)

        ## extract the patch and apply the rotation matrix
        X = affine_transform(self.img_array[image], rot_mat.T, offset=offset, output_shape=(ps_y, ps_x),
                             order=1) / 255.0
        ## extract the label
        Y = int(self.lbl_array[image, loc_y, loc_x] > 0)

        X = X.astype(np.float32)

        ## apply random flipping
        if self.rnd_flipping:
            X = self.rnd_flip(X)

        return X, Y

    def get_random_sample_from_class(self, label):
        ''' Extract a patch with a given label. '''

        # tip: use 'self.pos_idx' and 'self.neg_idx'
        if label == 0:
            idx = self.neg_idx[:, int(random.random() * len(self.neg_idx[0]))]
            image, loc_y, loc_x = idx[0], idx[1], idx[2]
        else:
            idx = self.pos_idx[:, int(random.random() * len(self.pos_idx[0]))]
            image, loc_y, loc_x = idx[0], idx[1], idx[2]

        return self.extract_sample(image, loc_y, loc_x)

#'''
#A convenience function to visualize the output of the patch extractor
#'''

#def visualize_samples(sampling_function, label):
#    for i in range(5):
#        X, Y = sampling_function(label)
#        plt.subplot(1,5,i+1)
#        plt.imshow(X, cmap='bone', vmin=0, vmax=1); plt.title('patch: ' + str(i) + " , class: " + str(Y))