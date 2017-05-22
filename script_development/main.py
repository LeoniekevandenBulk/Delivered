
import os
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import nibabel as nib

import os
import numpy as np
from PIL import Image
from math import cos, sin, radians, floor, ceil

import theano.tensor as T
import lasagne
import lasagne.layers as L

import random

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (40, 24)
matplotlib.rcParams['xtick.labelsize'] = 30

from BatchGenerator import BatchGenerator
from UNetClass import UNetClass
from Trainer import Trainer
from tools import get_file_list
from tools import output_size_for_input
from tools import show_slices
from tools import show_volumes
from tools import show_preprocessing
from tools import show_segmentation_prediction

import random

random.seed(0)

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
Split the 3D volumes in a training and validation set.
'''

nr_volumes = len(vol_batch)
vol_list = range(nr_volumes)
random.shuffle(vol_list)

validation_percentage = 0.3
nr_val_volumes = int(ceil(len(vol_batch)*validation_percentage))
nr_tra_volumes = nr_volumes - nr_val_volumes

# use the first images as validation
tra_list = vol_list[0:nr_tra_volumes]
val_list = vol_list[nr_tra_volumes:]

print("nr of training 3D volumes: " + str(len(tra_list)) + "\nnr of validation 3D volumes: " + str(len(val_list)))

'''
Set parameters
Set the parameters to suitable values.
Certain paramters have a large influence on the performance of your network, try to find the optimal parameters for
this problem!
'''

# UNet architecture
depth=5
branching_factor = 1

# Image dimensions
patch_size = (600,600)
out_size = output_size_for_input(patch_size, depth)
img_center = [256, 256]

# Training
learning_rate = 0.1
nr_epochs = 10
nr_batches = 1
batch_size = 20
max_rotation = 10
gaussian_blur = False
elastic_deformation = True
class_balancing = True
nr_validation_samples_per_3D_volume = 1
validation_batch_size = 1
liver_network_name = 'liver_network_LiTS'
lesion_network_name = 'lesion_network_LiTS'

'''
Creating the batch generators and testing the preprocessing
'''

trainBatchGenerator = BatchGenerator(augment=True) # (with augmentation)
valBatchGenerator = BatchGenerator(augment=False) # (without augmentation)

# Implement the get_random_sample_from_class() function in the class above. You can use the functions below to test if the output makes sense.

# Testing the preprocessing of the input image (patch_size) and output labels (out_size)
show_preprocess = False
if show_preprocess:
    nr_tests = 5 # 5 slices per test
    for i in range(nr_tests):
        show_preprocessing(nr_tra_volumes, train_batch_dir, trainBatchGenerator, patch_size, out_size, img_center)

'''
Create UNet networks
'''

ps_x = patch_size[0]
ps_y = patch_size[1]

# ## Compile the theano functions
inputs = T.ftensor4('X')
targets = T.itensor4('Y')
weights = T.ftensor4('W')

# Unet defined with pad='same' so input_size=output_size
# The original Unet is defined without padding to not create border effects in case of tiling a large image.
# Our images are not that large, and taken as a whole, so border effect are no concern here.

print("Liver network:")
liverNetwork = UNetClass(inputs, 
					    input_size=patch_size,
					    depth=depth,
					    branching_factor=branching_factor, # 2^6 filters for first level, 2^7 for second, etc.
					    num_input_channels=1,
					    num_classes=2,
					    pad='valid')

#outputs van liverNetwork worden inputs van lesionNetwork
print("Lesion network:")
lesionNetwork = UNetClass(inputs,
					    input_size=patch_size,
                        depth=depth,
                        branching_factor=branching_factor,  # 2^6 filters for first level, 2^7 for second, etc.
                        num_input_channels=1,
                        num_classes=2,
                        pad='valid')

print("Creating Theano training and validation functions for liver network ...")
liverNetwork.train_fn, liverNetwork.val_fn = liverNetwork.define_updates(inputs, targets, weights)
print("Creating Theano training and validation functions for lesion network ...")
lesionNetwork.train_fn, lesionNetwork.val_fn = lesionNetwork.define_updates(inputs, targets, weights)


'''
Training
'''

# plot learning curves
fig = plt.figure(figsize=(30, 15))
plt.xlabel('epoch', size=40)
plt.ylabel('loss', size=40)
fig.labelsize=40

liverTrainer = Trainer(liverNetwork, "liverNet", tra_list, val_list, learning_rate=learning_rate, patch_size=patch_size)
lesionTrainer = Trainer(lesionNetwork, "lesionNet", tra_list, val_list, learning_rate=learning_rate, patch_size=patch_size)

# Main training loop
tra_loss_lst = []
val_loss_lst = []
tra_acc_lst = []
val_acc_lst = []
best_val_acc = 0
for epoch in range(nr_epochs):
    print('Epoch {}/{}'.format(epoch + 1,nr_epochs))

    tra_losses = []
    tra_accs = []
    val_losses = []
    val_accs = []
    for batch in range(nr_batches):
        print('Batch {}/{}'.format(batch + 1, nr_batches))
        # Training
        X_tra, Y_tra = trainBatchGenerator.get_batch(tra_list, train_batch_dir, batch_size,
                                     patch_size, out_size, img_center, group_labels=((0,1),2), group_percentages=(0.5,0.5))
        loss, l2_loss, accuracy, target_prediction, prediction = lesionTrainer.train_batch(X_tra, Y_tra, verbose=True)
        print 'training loss, accuracy', loss, accuracy
        tra_losses.append(loss)
        tra_accs.append(accuracy)
        # Validation
        X_val, Y_val = valBatchGenerator.get_batch(val_list, train_batch_dir, batch_size,
                                     patch_size, out_size, img_center, group_labels=((0,1),2), group_percentages=(0.5,0.5))
        loss, l2_loss, accuracy, target_prediction, prediction = lesionTrainer.validate_batch(X_val, Y_val, verbose=True)
        print 'validation loss, accuracy', loss, accuracy
        val_losses.append(loss)
        val_accs.append(accuracy)

    tra_loss_lst.append(np.mean(tra_losses))
    tra_acc_lst.append(np.mean(tra_accs))
    val_loss_lst.append(np.mean(val_losses))
    val_acc_lst.append(np.mean(val_accs))

    if np.mean(val_accs) > best_val_acc:
        best_val_acc = np.mean(val_accs)
        # save networks
        params = L.get_all_param_values(liverNetwork.net)
        np.savez(os.path.join('./', liver_network_name + '.npz'), params=params)
        params = L.get_all_param_values(lesionNetwork.net)
        np.savez(os.path.join('./', lesion_network_name + '.npz'), params=params)

    # plot learning curves
    fig = plt.figure(figsize=(10, 5))
    tra_loss_plt, = plt.plot(range(len(tra_loss_lst)), tra_loss_lst, 'b')
    val_loss_plt, = plt.plot(range(len(val_loss_lst)), val_loss_lst, 'g')
    tra_acc_plt, = plt.plot(range(len(tra_acc_lst)), tra_acc_lst, 'm')
    val_acc_plt, = plt.plot(range(len(val_acc_lst)), val_acc_lst, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend([tra_loss_plt, val_loss_plt, tra_acc_plt, val_acc_plt],
               ['training loss', 'validation loss', 'training accuracy', 'validation accuracy'],
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Best validation accuracy = {:.2f}%'.format(100. * best_val_acc), size=40)
    plt.show(block=False)
    plt.pause(.5)

plt.legend([tra_loss_plt, val_loss_plt, tra_acc_plt, val_acc_plt],
           ['training loss', 'validation loss', 'training accuracy', 'validation accuracy'],
           loc='center left', bbox_to_anchor=(1, 0.5))

'''
Visualize some validation segmentations
'''

show_segmentations = True
if show_segmentations:
    show_segmentation_prediction(valBatchGenerator, lesionTrainer, val_list, train_batch_dir,
                                 patch_size, out_size, img_center)

'''
Evaluate the performance on the test set and submit to Challenger

We can now apply the trained fully convolutional network on our independent test set.
'''

# Load the images:

# test images
test_dir='../data/Test-Data'

vol_test = sorted(get_file_list(test_dir, 'test-volume')[0])

result_output_folder = os.path.join(test_dir, 'results')
if not (os.path.exists(result_output_folder)):
    os.mkdir(result_output_folder)

# ## Apply the trained network to the images

read_network_parameters = False
if read_network_parameters:
    liver_network_name = 'liver_network_LiTS'
    lesion_network_name = 'lesion_network_LiTS'
    # initialize the networks used in this experiment (this may change)
    liverNetwork = UNetClass(inputs, input_size=patch_size,
                             depth=5,
                             branching_factor=1,  # 2^6 filters for first level, 2^7 for second, etc.
                             num_input_channels=1,
                             num_classes=2,
                             pad='same')
    npz = np.load('./' + network_name + '.npz')  # load stored parameters
    lasagne.layers.set_all_param_values(liverNetwork.net, npz['params'])  # set parameters
    lesionNetwork = UNetClass(inputs, input_size=patch_size,
                              depth=5,
                              branching_factor=1,  # 2^6 filters for first level, 2^7 for second, etc.
                              num_input_channels=1,
                              num_classes=2,
                              pad='same')
    npz = np.load('./' + network_name + '.npz')  # load stored parameters
    lasagne.layers.set_all_param_values(lesionNetwork.net, npz['params'])  # set parameters

print("Creating Theano predict function ...")
lesionNetwork.predict_fn = lesionNetwork.define_predict(inputs)

# one forward pass on the test set
threshold = 0.5
print ('predicting...')
for i, vol in enumerate(vol_test):
    # load 3D volumes
    vol_array = nib.load(vol).get_data()
    vol_array=vol_array[:,:,vol_array.shape[2]/2:vol_array.shape[2]/2+1] # testing middle slice
    # organize test slices one slice per batch
    X = np.zeros((1, 1, vol_array.shape[0], vol_array.shape[1]))

    # get every slice
    thresholded_image = np.zeros(vol_array.shape)
    for slice in range(vol_array.shape[2]):
        print ('predicting slice '+str(slice)+'/'+str(vol_array.shape[2]-1)+', test volume '+vol)
        X[0, 0, :, :] = vol_array[:ps_y, :ps_x, slice]

        # predict softmax proabbilities
        probability = lesionNetwork.predict_fn(X.astype(np.float32))

        #probability[0] is for no label, probability[1] is for label
        probability_labeled = np.asarray(probability)[:,:,1].reshape(patch_size)

        # binary version by thresholding
        thresholded_image[:, :, slice] = probability_labeled > threshold

    # save output to file
    np.save(vol.replace('volume', 'lesions'), thresholded_image)







