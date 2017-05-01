
import os
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.testing import data_path

#import nilearn
#from nilearn import plotting

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
from IPython import display
import time
#from tqdm import tnrange, tqdm_notebook

#from SampleExtractor import visualize_samples
from SampleExtractor import SampleExtractor
from BatchExtractor import BatchExtractor
from fcn import build_network
from fcn import softmax
from fcn import softmax_deterministic
from fcn import training_function
from fcn import validate_function
from fcn import evaluate_function
from tools import get_file_list
from tools import show_image

#plt.isinteractive()

'''
Data loading and visualization
'''
data_path='./Training_Batch_1'
vol_filename = os.path.join(data_path, 'volume-10.nii')
vol = nib.load(vol_filename)
vol_array = vol.get_data()

seg_filename = os.path.join(data_path, 'segmentation-10.nii')
seg = nib.load(seg_filename)
seg_array = seg.get_data()

#plotting.plot_glass_brain('segmentation-0.nii')
#plotting(n1_img)

# Set data_dir to the location of the folder containing the DRIVE data
data_dir = "../Assignments/assignment_7/"

tra_img_dir = os.path.join(data_dir, 'training', 'images')
tra_msk_dir = os.path.join(data_dir, 'training', 'mask')
tra_lbl_dir = os.path.join(data_dir, 'training', '1st_manual')

tra_imgs_all = sorted(get_file_list(tra_img_dir, 'tif')[0])
tra_msks_all = sorted(get_file_list(tra_msk_dir, 'gif')[0])
tra_lbls_all = sorted(get_file_list(tra_lbl_dir, 'gif')[0])

'''
Split the data in a training and validation set.
'''

validation_percentage = 0.3
n_validation_images = int(ceil(len(tra_imgs_all)*validation_percentage))

# use the first images as validation
val_imgs = tra_imgs_all[0:n_validation_images]
val_msks = tra_msks_all[0:n_validation_images]
val_lbls = tra_lbls_all[0:n_validation_images]

# the rest as training
tra_imgs = tra_imgs_all[n_validation_images:]
tra_msks = tra_msks_all[n_validation_images:]
tra_lbls = tra_lbls_all[n_validation_images:]

print("nr of training images: " + str(len(tra_imgs)) + "\nnr of validation images: " + str(len(val_imgs)))

# Show the training set using the function defined
#for i in range(len(tra_imgs)):
#    show_image(i, tra_imgs, tra_msks, tra_lbls)

'''
Creating and testing the basic random sample extractor
'''

# Implement the get_random_sample_from_class() function in the class above. You can use the functions below to test if the output makes sense.

# Here we test the patch extractor without random rotation and without flipping.

# define a random sample extractor object
random_sample_extractor = SampleExtractor((31,31),tra_imgs, tra_msks, tra_lbls, max_rotation=0, rnd_flipping=False)

# Execute the following cells several times to see if the results you get make sense.

# visualize examples of background (label = 0)
#visualize_samples(random_sample_extractor.get_random_sample_from_class, 0)
for i in range(5):
    X, Y = random_sample_extractor.get_random_sample_from_class(0)
    plt.subplot(1, 5, i + 1)
    plt.imshow(X, cmap='bone', vmin=0, vmax=1)
    plt.title('patch: ' + str(i) + " , class: " + str(Y))

# visualize example of vessels (label = 1)
#visualize_samples(random_sample_extractor.get_random_sample_from_class, 1)
for i in range(5):
    X, Y = random_sample_extractor.get_random_sample_from_class(1)
    plt.subplot(1, 5, i + 1)
    plt.imshow(X, cmap='bone', vmin=0, vmax=1)
    plt.title('patch: ' + str(i) + " , class: " + str(Y))

# ## Implement rotation augmentation
#
# Implement the ```get_rnd_rotation()``` function in the class above.
# You can use the functions below to test if the output makes sense.

#Now we enable rotation, and test it again.
random_sample_extractor = SampleExtractor((31,31),tra_imgs, tra_msks, tra_lbls, max_rotation=15, rnd_flipping=False)

# visualize example of vessels (label = 1)
for i in range(5):
    X, Y = random_sample_extractor.get_random_sample_from_class(1)
    plt.subplot(1, 5, i + 1)
    plt.imshow(X, cmap='bone', vmin=0, vmax=1)
    plt.title('patch: ' + str(i) + " , class: " + str(Y))

# visualize example of background (label = 0)
for i in range(5):
    X, Y = random_sample_extractor.get_random_sample_from_class(0)
    plt.subplot(1, 5, i + 1)
    plt.imshow(X, cmap='bone', vmin=0, vmax=1)
    plt.title('patch: ' + str(i) + " , class: " + str(Y))

# ## Implement mirroring augmentation
# Implement the ```rnd_flip()`` function in the class above.
# You can use the functions below to test if the output makes sense.

random_sample_extractor = SampleExtractor((31,31),tra_imgs, tra_msks, tra_lbls, max_rotation=0, rnd_flipping=True)

# visualize example of vessels (label = 1)
for i in range(5):
    X, Y = random_sample_extractor.get_random_sample_from_class(1)
    plt.subplot(1, 5, i + 1)
    plt.imshow(X, cmap='bone', vmin=0, vmax=1)
    plt.title('patch: ' + str(i) + " , class: " + str(Y))

# visualize example of background (label = 0)
for i in range(5):
    X, Y = random_sample_extractor.get_random_sample_from_class(0)
    plt.subplot(1, 5, i + 1)
    plt.imshow(X, cmap='bone', vmin=0, vmax=1)
    plt.title('patch: ' + str(i) + " , class: " + str(Y))

'''
Set parameters
Set the parameters to suitable values.
Certain paramters have a large influence on the performance of your network, try to find the optimal parameters for
this problem!
'''

learning_rate = 0.01
nr_epochs = 10
nr_iterations_per_epoch = 1000
patch_size = (31,31)
batch_size = 100
max_rotation = 15
flipping = True
class_balancing = True
nr_validation_samples = 1000
validation_batch_size = 100
network_name = 'network_ass7'


# ## Compile the theano functions
inputs = T.ftensor4('X')
targets = T.ftensor4('Y')

network = build_network(inputs)

train_fn = training_function(network=network, input_tensor=inputs, target_tensor=targets, learning_rate=learning_rate,
                             use_l2_regularization=False, l2_lambda=0.000001)
validation_fn = validate_function(network=network, input_tensor=inputs, target_tensor=targets)
evaluation_fn = evaluate_function(network=network, input_tensor=inputs)

'''
Create the
 - sample extractors
 - batch extractors
 - constant validation set
 - Initialize all monitoring statistics
'''

random.seed(0)

sample_ex_train = SampleExtractor(patch_size, tra_imgs, tra_msks, tra_lbls, max_rotation=max_rotation, rnd_flipping=flipping)
batch_ex_train = BatchExtractor(sample_ex_train, batch_size, class_balancing=class_balancing)

sample_ex_val = SampleExtractor(patch_size, val_imgs, val_msks, val_lbls, max_rotation=max_rotation, rnd_flipping=flipping)
batch_ex_val = BatchExtractor(sample_ex_val, nr_validation_samples, class_balancing=class_balancing)

validation_X, validation_Y = batch_ex_val.get_random_batch_balanced()

tra_loss_lst = []
val_loss_lst = []
tra_acc_lst = []
val_acc_lst = []
best_val_acc = 0

'''
Main training loop

Now that we have a pipeline for creating batches on the fly, we can now train our fully convolutional network!
The following code trains for a certain amount of time, defined in the parameters section above, and will then evaluate
the performance on our validation set. After plotting the results, the next epoch will start.
Try to optimize the parameters to get the best performance possible! (>90% accuracy should be possible!)
'''

# plot learning curves
fig = plt.figure(figsize=(30, 15))
plt.xlabel('epoch', size=40)
plt.ylabel('loss', size=40)
fig.labelsize=40
# Main training loop
for epoch in range(nr_epochs):
    print('Epoch {}'.format(epoch + 1))
    # training
    tra_losses = []
    tra_accs = []
    print('training...')
    for b in range(0, nr_iterations_per_epoch),:
        X, Y = batch_ex_train.get_random_batch_balanced()
        # print 'train X', X.shape, X.dtype, X.min(), X.max(), np.any(np.isnan(X))
        # print 'train Y', Y.shape, Y.dtype, Y.min(), Y.max(), np.any(np.isnan(Y))
        loss, accuracy = train_fn(X.astype(np.float32), Y.astype(np.float32))
        # print 'train loss, accuracy', loss, accuracy
        tra_losses.append(loss)
        tra_accs.append(accuracy)
    tra_loss_lst.append(np.mean(tra_losses))
    tra_acc_lst.append(np.mean(tra_accs))

    # validation
    val_losses = []
    val_accs = []
    print('validation...')
    # print validation_batch_size, nr_validation_samples
    # print validation_X.shape, validation_Y.shape

    for b in range(0, nr_validation_samples, validation_batch_size):
        X = validation_X[b:min(b + validation_batch_size, nr_validation_samples)]
        Y = validation_Y[b:min(b + validation_batch_size, nr_validation_samples)]
        # print 'test X', X.shape, X.dtype#, X.min(), X.max(), np.any(np.isnan(X))
        # print 'test Y', Y.shape, Y.dtype, Y.min(), Y.max(), np.any(np.isnan(Y))
        loss, accuracy = validation_fn(X.astype(np.float32), Y.astype(np.float32))
        # print 'test loss, accuracy', loss, accuracy
        val_losses.append(loss)
        val_accs.append(accuracy)
    val_loss_lst.append(np.mean(val_losses))
    val_acc_lst.append(np.mean(val_accs))
    # print val_accs
    # continue
    if np.mean(val_accs) > best_val_acc:
        best_val_acc = np.mean(val_accs)
        # save network
        params = L.get_all_param_values(network)
        np.savez(os.path.join('./', network_name + '.npz'), params=params)

    # plot learning curves
    #fig = plt.figure(figsize=(10, 5))
    tra_loss_plt, = plt.plot(range(len(tra_loss_lst)), tra_loss_lst, 'b')
    val_loss_plt, = plt.plot(range(len(val_loss_lst)), val_loss_lst, 'g')
    tra_acc_plt, = plt.plot(range(len(tra_acc_lst)), tra_acc_lst, 'm')
    val_acc_plt, = plt.plot(range(len(val_acc_lst)), val_acc_lst, 'r')
    #plt.xlabel('epoch')
    #plt.ylabel('loss')
    #plt.legend([tra_loss_plt, val_loss_plt, tra_acc_plt, val_acc_plt],
    #           ['training loss', 'validation loss', 'training accuracy', 'validation accuracy'],
    #           loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Best validation accuracy = {:.2f}%'.format(100. * best_val_acc), size=40)
    plt.show(block=False)
    plt.pause(.5)

plt.legend([tra_loss_plt, val_loss_plt, tra_acc_plt, val_acc_plt],
           ['training loss', 'validation loss', 'training accuracy', 'validation accuracy'],
           loc='center left', bbox_to_anchor=(1, 0.5))

'''
Evaluate the performance on the test set and submit to Challenger

We can now apply the trained fully convolutional network on our independent test set.
'''

# Load the images:

# test images
tes_img_dir = os.path.join(data_dir, 'test', 'images')
tes_msk_dir = os.path.join(data_dir, 'test', 'mask')

tes_imgs = sorted(get_file_list(tes_img_dir, 'tif')[0])
tes_msks = sorted(get_file_list(tes_msk_dir, 'gif')[0])

result_output_folder = os.path.join(data_dir, 'test', 'results')
if not (os.path.exists(result_output_folder)):
    os.mkdir(result_output_folder)

# ## Apply the trained network to the images

# indicate the name of the network for this test
network_name = 'network_ass7'

# initialize the network used in this experiment (this may change)
network = build_network(inputs)
npz = np.load('./' + network_name + '.npz')  # load stored parameters
lasagne.layers.set_all_param_values(network, npz['params'])  # set parameters

# one forward pass on the test set
fig, axes = plt.subplots(nrows=len(tes_imgs), ncols=3, figsize=(15, len(tes_imgs)*5))
threshold = 0.5
for f in range(len(tes_imgs)):
    # open image
    img = np.asarray(Image.open(tes_imgs[f]))

    # extract green channel and normalize
    img_g = img[:, :, 1].squeeze().astype(float) / 255.0

    # zero padding to apply fully-convolutional network
    img_g_padded = np.pad(img_g, ((patch_size[0] // 2, patch_size[0] // 2),
                                  (patch_size[1] // 2, patch_size[1] // 2)), 'constant', constant_values=[0, 0])

    # forward pass
    probability = evaluation_fn(np.expand_dims(np.expand_dims(img_g_padded.astype(np.float32), axis=0), axis=0))
    probability = probability[0, 1, :, :]

    # binary version by thresholding
    thresholded_image = probability > threshold

    # show results
    ax0 = plt.subplot2grid((len(tes_imgs), 3), (f, 0))
    ax1 = plt.subplot2grid((len(tes_imgs), 3), (f, 1))
    ax2 = plt.subplot2grid((len(tes_imgs), 3), (f, 2))
    #plt.subplot(1, 3, 1)
    ax0.imshow(img_g_padded, cmap='gray')
    #plt.subplot(1, 3, 2)
    ax1.imshow(probability, cmap='jet')
    #plt.subplot(1, 3, 3)
    ax2.imshow(thresholded_image, cmap='gray')

plt.show()





