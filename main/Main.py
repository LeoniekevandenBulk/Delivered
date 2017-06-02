import numpy as np
np.set_printoptions(precision=2, suppress=True)
import theano.tensor as T

from math import ceil
from Trainer import *
from tools import *

import random

from BatchGenerator import BatchGenerator
from BatchAugmenter import BatchAugmenter

import time
import cProfile
import re

start_time = time.time()

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

# Determine whether to test or not
run_test = False

# UNet architecture
depth = 5
branching_factor = 2 # 2^6 filters for first level, 2^7 for second, etc.

# Image dimensions
patch_size = (650,650)
out_size = output_size_for_input\
    (patch_size, depth)
img_center = [256, 256]

# Training
learning_rate = 0.1
nr_epochs = 10 # 10
nr_train_batches = 10 # 10
nr_val_batches = 2 # 2
batch_size = 5

max_rotation = 10
liver_aug_params = [0.1,0.0,0.9]
lesion_aug_params = [0.1,0.0,0.6]

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
trainer = Trainer(SURFsara)

# Loop-booleans to allow specific training
train_liver = True
train_lesion = True

if train_liver:
    # Load or train liver segmentation network
    print("Liver Network...")

    if (load_liver_segmentation):
        liver_network = trainer.readNetwork(liver_segmentation_name, patch_size,
                inputs, targets, weights, depth, branching_factor)
        liver_threshold = 0.5 # Just to catch potential errors. We should know this during runtime
    else:
        mask = 'none'
        liver_network, liver_threshold = trainer.trainNetwork(start_time, liver_segmentation_name, mask,
                patch_size, depth, branching_factor, out_size, img_center,
                train_batch_dir, inputs, targets, weights,
                "liver", tra_list, val_list,
                liver_aug_params, learning_rate,
                nr_epochs, nr_train_batches, nr_val_batches, batch_size)

if train_lesion:
    # Load or train lesion detection network
    print("Lesion Network...")
    if (load_lesion_detection):
        lesion_network = trainer.readNetwork(lesion_detection_name, patch_size,
                inputs, targets, weights, depth, branching_factor)
    else:
        mask = 'ground truth'  # choose 'liver' or 'ground_truth'

        # Set relevant variables depending on whether liver network was trained
        if train_liver:
            mask_network = liver_network
        
        else:
            mask_network = None
            liver_threshold = 0.5
        
        lesion_network, lesion_threshold = trainer.trainNetwork(start_time, lesion_detection_name, mask,
                patch_size, depth, branching_factor, out_size, img_center,
                train_batch_dir, inputs, targets, weights,
                "lesion", tra_list, val_list,
                lesion_aug_params, learning_rate,
                nr_epochs, nr_train_batches, nr_val_batches, batch_size,
                mask_network, threshold = liver_threshold)

'''
Move on to testing and submitting results (asks every time)
'''
if (run_test):
    print("Testing now...")

    tester = Tester(SURFsara)

    # Load test data from folder
    test_dir='../data/Test-Data'
    vol_test = sorted(get_file_list(test_dir, 'test-volume')[0])

    # Make sure that the output folder exists
    result_output_folder = os.path.join(test_dir, 'results')
    if not (os.path.exists(result_output_folder)):
        os.mkdir(result_output_folder)

    for i, vol in enumerate(vol_test):
        # Load 3D volumes
        vol_array = nib.load(vol).get_data()
        vol_array=vol_array[:,:,vol_array.shape[2]/2:vol_array.shape[2]/2+1] # testing middle slice
        
        # Match input size with expected array dimension
        input_size = liver_network.input_size
        X = np.zeros((1, 1, input_size[0], input_size[1]))

        # Classify each slice
        classification = np.zeros(vol_array.shape)
        for j in range(vol_array.shape[2]):

            print ('predicting slice '+str(img_slice)+'/'+str(vol_array.shape[2]-1)+', test volume '+vol)

            # Copy slice into memory
            img_slice = vol_array[:, :, j]

            # Normalize values of the image slice
            img_slice = np.clip(img_slice, -200, 300)
            #X_tra = (X_tra - X_tra.mean()) / X_tra.std()
            img_slice = (img_slice + 200)/ 500

            # PAD X FOR LIVER DETECTION
            img_slice = 

            
            # Put image slice into X
            X[0, 0, :, :] = img_slice

            # Apply liver segmentation network
            liver_seg_mask = liver_network.predict_fn(X.astype(np.float32))        
            # Turn heatmap into binary classification
            liver_seg_mask = (liver_seg_mask > liver_threshold).astype(np.int32)


            # PAD LIVER MASK FOR LESION SEGMENTATION

            
            # Apply liver mask to slice
            X[0,0,:,:] = np.multiply(X[0, 0, :,:], liver_seg_mask)

            # Apply lesion detection network
            lesion_detect = lesion_network.predict_fn(X.astype(np.float32))
            # Turn heatmap into binary classification
            lesion_detect = (lesion_detect > lesion_threshold).astype(np.int32)        
            # Match format (lesion has value 2)
            lesion_detect = lesion_detect * 2

            # PAD LESION DETECTION FOR 512x512 CRITERIUM
            

            # Then save into classification array
            classification[:,:, img_slice] = lesion_detect

        # Turn image into .nii file

        # Save output to file
        np.save("test-segmentation-{}.nii".format(i), classification)








