import numpy as np
np.set_printoptions(precision=2, suppress=True)
import theano.tensor as T
import nibabel as nib

from math import ceil
from Trainer import *
from tools import *

import random

from BatchGenerator import BatchGenerator
from BatchAugmenter import BatchAugmenter

import time
import cProfile
import re
import os

start_time = time.time()

# Seed random to make sure validation set is always same
random.seed(0)

'''
Set parameters to suitable values
'''

# Boolean to catch SURFsara dependent code
SURFsara = True

# Loop-booleans to allow specific training
train_liver = True
train_lesion = False
train_lesion_only = False # If put to True also put train_lesion to True!

# Variables that define which network to load from file (or not)
liver_segmentation_name = 'liver_network_LiTS_0.195079_0.939129878833_0.632622622623'
load_liver_segmentation = False

lesion_detection_name = 'lesion_network_LiTS'
load_lesion_detection = False

# Determine whether to test or not
run_test = False

# Read slices from file, names of the files to read slices from
read_slices = False
vol_tra_slices_name = 'vol_tra_slices.npy' #vol_tra_slices
seg_tra_slices_name = 'seg_tra_slices.npy' #seg_tra_slices
msk_tra_slices_name = 'msk_tra_slices.npy' #msk_tra_slices
vol_val_slices_name = 'vol_val_slices.npy' #vol_val_slices
seg_val_slices_name = 'seg_val_slices.npy' #seg_val_slices
msk_val_slices_name = 'msk_val_slices.npy' #msk_val_slices
slice_files = np.array([vol_tra_slices_name, seg_tra_slices_name, msk_tra_slices_name,
                        vol_val_slices_name, seg_val_slices_name, msk_val_slices_name])
nr_slices_per_volume=50

# Plotting
show_segmentation_predictions = True

# Save network parameters every epoch (normally only saved when validation loss improved)
save_network_every_epoch = True

# UNet architecture
depth = 5
branching_factor = 6 # 2^6 filters for first level, 2^7 for second, etc.

# Image dimensions
patch_size = (690,690)
out_size = output_size_for_input\
    (patch_size, depth)
img_center = [256, 256]

# Training
learning_rate = 0.001
nr_epochs = 15 # 10
batch_size = 2
group_percentages = (0.3, 0.7)

weight_balance_liver = 4
weight_balance_lesion = 10

max_rotation = 10
liver_aug_params = [0.2,0.2,0.2]
lesion_aug_params = [0.2,0.2,0.2]


# Theano tensors
inputs = T.ftensor4('X')
targets = T.itensor4('Y')
weights = T.ftensor4('W')

'''
Print important variables/settings
'''
print_settings(train_liver, train_lesion, train_lesion_only,
                load_liver_segmentation, liver_segmentation_name,
                read_slices, nr_slices_per_volume,
                show_segmentation_predictions,
                save_network_every_epoch,
                depth, branching_factor,
                patch_size, out_size, img_center,
                learning_rate, nr_epochs, batch_size, group_percentages,
                weight_balance_liver, weight_balance_lesion,
                max_rotation, liver_aug_params, lesion_aug_params )

'''
Train data loading
'''

train_batch_dir='../data/Training_Batch'

vol_batch = sorted(get_file_list(train_batch_dir, 'volume')[0])
seg_batch = sorted(get_file_list(train_batch_dir, 'segmentation')[0])


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
trainer = Trainer(SURFsara, save_network_every_epoch)

if train_liver:
    # Load or train liver segmentation network
    print("Liver Network...")
    if (load_liver_segmentation):
        liver_network = trainer.readNetwork(liver_segmentation_name, patch_size,
                inputs, targets, weights, depth, branching_factor)
        liver_name = liver_segmentation_name.split('_')
        liver_threshold = float(liver_name[-1])
    else:
        mask_name = 'none'
        
        liver_network, liver_threshold = trainer.trainNetwork(start_time, liver_segmentation_name,
                patch_size, depth, branching_factor, out_size, img_center,
                train_batch_dir, inputs, targets, weights, "liver", tra_list, val_list,
                liver_aug_params, learning_rate, nr_epochs, batch_size, group_percentages, read_slices,
                slice_files, nr_slices_per_volume, weight_balance_liver)
    if show_segmentation_predictions:
        show_segmentation_prediction(trainer, liver_network, liver_threshold, val_list, train_batch_dir,
                                     patch_size, out_size, img_center, "liver", read_slices, slice_files, 
				     nr_slices_per_volume, weight_balance_liver, mask_name=None, mask_network=None)

if train_lesion:
    # Load or train lesion detection network
    print("Lesion Network...")
    if (load_lesion_detection):
        lesion_network = trainer.readNetwork(lesion_detection_name, patch_size,
                inputs, targets, weights, depth, branching_factor)
        lesion_name = lesion_detection_name.split('_')
        lesion_threshold = float(lesion_name[-1])
    else:

        # Set relevant variables depending on whether liver network was trained
        if train_liver:
            mask_network = liver_network
            mask_name = "liver_network"
        elif train_lesion_only:
            mask_network = None
            liver_threshold = 0.5
            mask_name = None 
        else:
            mask_network = None
            liver_threshold = 0.5
            mask_name = 'ground_truth'  # choose 'liver' or 'ground_truth'

        lesion_network, lesion_threshold = trainer.trainNetwork(start_time, lesion_detection_name,
                patch_size, depth, branching_factor, out_size, img_center, train_batch_dir, inputs,
                targets, weights, "lesion", tra_list, val_list, lesion_aug_params, learning_rate,
                nr_epochs, batch_size, group_percentages, read_slices, slice_files, nr_slices_per_volume,
                weight_balance_lesion, mask_network, mask_name, mask_threshold = liver_threshold)
    if show_segmentation_predictions:
        show_segmentation_prediction(trainer, liver_network, liver_threshold, val_list, train_batch_dir,
                                     patch_size, out_size, img_center, "liver", read_slices, slice_files, 
				     nr_slices_per_volume, weight_balance_lesion, mask_name=None, mask_network=None)

'''
Move on to testing and saving results
'''
if (run_test):
    print("Testing now...")
    
    # Folder location
    test_dir='../data/LITS-Challenge-Test-Data'
    test_list = sorted(get_file_list(test_dir, 'test-volume')[0])

    # Load test data from folder
    tester = Tester(patch_size, out_size, liver_network, liver_threshold, lesion_network, lesion_threshold)

    # Get test results (segmentations are saved in results folder)
    tester.perform_test(test_list, test_dir)
