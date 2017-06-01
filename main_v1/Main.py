import numpy as np
np.set_printoptions(precision=2, suppress=True)
import theano.tensor as T

from math import ceil
from Trainer import *
from tools import *

import random

# Seed random to make sure validation set is always same
random.seed(0)


'''
Set parameters to suitable values
'''

# Boolean to catch SURFsara dependent code
SURFsara = False

# Variables that define which network to load from file (or not)
liver_segmentation_name = 'liver_network_LiTS'
load_liver_segmentation = False

lesion_detection_name = 'lesion_network_LiTS'
load_lesion_detection = False

# UNet architecture
depth = 5
branching_factor = 2  # 2^6 filters for first level, 2^7 for second, etc.

# Image dimensions
patch_size = (650,650)
out_size = output_size_for_input(patch_size, depth)
img_center = [256, 256]

# Training
learning_rate = 0.1
nr_epochs = 1
nr_train_batches = 1
nr_val_batches = 1
batch_size = 5

max_rotation = 10
liver_aug_params = [0.1,0.8,0.9]
lesion_aug_params = [0.1,0.8,0.6]

liver_network_name = 'liver_network_LiTS'
lesion_network_name = 'lesion_network_LiTS'

# Theano tensors
if SURFsara:
    inputs = T.dtensor4('X')
    targets = T.itensor4('Y')
    weights = T.dtensor4('Z')
else:
    inputs = T.ftensor4('X')
    targets = T.itensor4('Y')
    weights = T.ftensor4('W')



'''
Data loading
'''

train_batch_dir='../data/Training_Batch'

vol_batch = sorted(get_file_list(train_batch_dir, 'volume')[0])
seg_batch = sorted(get_file_list(train_batch_dir, 'segmentation')[0])

show_vol = False
if show_vol:
        show_volumes(vol_batch, seg_batch)


'''
Split the 3D volumes in a training and validation set
'''

nr_volumes = len(vol_batch)
vol_list = range(nr_volumes)
random.shuffle(vol_list)

validation_percentage = 0.3
nr_val_volumes = int(ceil(nr_volumes*validation_percentage))
nr_tra_volumes = nr_volumes - nr_val_volumes

# Use the first images as validation
tra_list = vol_list[0:nr_tra_volumes]
val_list = vol_list[nr_tra_volumes:]

print("nr of training 3D volumes: " + str(len(tra_list)) + "\nnr of validation 3D volumes: " + str(len(val_list)))


'''
Initiate training/loading of networks
'''

# Create class to train (or load) networks
trainer = Trainer(SURFsara)


# Load or train liver segmentation network
print("Liver Network...")
if (load_liver_segmentation):
    liver_network = trainer.readNetwork(liver_segmentation_name, patch_size, 
            inputs, targets, weights, depth, branching_factor)
    liver_threshold = 0.5 # Just to catch potential errors. We should know this during runtime
else:
    liver_network, liver_threshold = trainer.trainNetwork(liver_segmentation_name,
            patch_size, depth, branching_factor, out_size, img_center,
            train_batch_dir, inputs, targets, weights, 
            "liver", tra_list, val_list,
            liver_aug_params, learning_rate,
            nr_epochs, nr_train_batches, nr_val_batches, batch_size)


# Load or train lesion detection network
print("Lesion Network...")
if (load_lesion_detection):
    lesion_network = trainer.readNetwork(lesion_detection_name, patch_size,
            inputs, targets, weights, depth, branching_factor)
else:
    lesion_network, lesion_threshold = trainer.trainNetwork(lesion_detection_name,
            patch_size, depth, branching_factor, out_size, img_center,
            train_batch_dir, inputs, targets, weights, 
            "lesion", tra_list, val_list,
            lesion_aug_params, learning_rate,
            nr_epochs, nr_train_batches, nr_val_batches, batch_size,
            mask_network = liver_network, threshold = liver_threshold)












