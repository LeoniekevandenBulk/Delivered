
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

from SampleExtractor import SampleExtractor
from BatchExtractor import BatchExtractor
import UNet
from tools import get_file_list

#plt.isinteractive()

'''
Data loading and visualization
'''
def show_slices(vol_slices, seg_slices):
    """ Function to display row of image slices """
    nslice = len(vol_slices)
    fig, axes = plt.subplots(2, nslice)
    for i, slice in enumerate(vol_slices):
        axes[0, i].imshow(slice.T, cmap="gray", origin="lower")
    for i, slice in enumerate(seg_slices):
        axes[1, i].imshow(slice.T, cmap="gray", origin="lower", clim=(0, 2))

train_batch_dir='./data/Training_Batch_1'

vol_batch = sorted(get_file_list(train_batch_dir, 'volume')[0])
seg_batch = sorted(get_file_list(train_batch_dir, 'segmentation')[0])

show_volumes = False
if show_volumes:
    for i, (vol, seg) in enumerate(zip(vol_batch, seg_batch)):
        #i = 0
        #vol = vol_batch_1[i]
        #seg = seg_batch_1[i]
        #vol_filename = vol_batch_1[i]
        vol_proxy = nib.load(vol)
        vol_array = vol_proxy.get_data()
        #seg_filename = seg_batch_1[9]
        seg_proxy = nib.load(seg)
        seg_array = seg_proxy.get_data()
        name=vol.split('-')[1].split('.')[0]
        mid_0 = int(vol_array.shape[0]/2)
        mid_1 = int(vol_array.shape[1]/2)
        mid_2 = int(vol_array.shape[2]/2)
        vol_slice_0 = vol_array[mid_0, :, :]
        vol_slice_1 = vol_array[:, mid_1, :]
        vol_slice_2 = vol_array[:, :, mid_2]
        seg_slice_0 = seg_array[mid_0, :, :]
        seg_slice_1 = seg_array[:, mid_1, :]
        seg_slice_2 = seg_array[:, :, mid_2]
        show_slices([vol_slice_0, vol_slice_1, vol_slice_2], [seg_slice_0, seg_slice_1, seg_slice_2])
        plt.suptitle("Center slices for 3D volume "+vol, size=40)
        plt.show(block=False)
        plt.pause(0.5)

'''
Creating and testing the basic random sample extractor
'''

# Implement the get_random_sample_from_class() function in the class above. You can use the functions below to test if the output makes sense.

# Here we test the patch extractor without random rotation and without flipping.
test_sample_extractor = False
if test_sample_extractor:
    # define a random sample extractor object for the first 3D volume
    random_sample_extractor = SampleExtractor((256,512), vol_batch[0], seg_batch[0], labeling='liver', max_rotation=0, gaussian_blur=False, elastic_deformation=False)
    # Execute the following cells several times to see if the results you get make sense.
    X_0 = []
    Y_0 = []
    for i in range(5):
        X, Y = random_sample_extractor.get_random_sample_from_class(0)
        X_0.append(X)
        Y_0.append(Y)
    show_slices(X_0, Y_0)
    plt.suptitle("Label=0 slices: no liver", size=40)
    plt.show(block=False)
    X_1 = []
    Y_1 = []
    for i in range(5):
        X, Y = random_sample_extractor.get_random_sample_from_class(1)
        X_1.append(X)
        Y_1.append(Y)
    show_slices(X_1, Y_1)
    plt.suptitle("Label=1 slices: liver without lesions", size=40)
    plt.show(block=False)
    X_2 = []
    Y_2 = []
    for i in range(5):
        X, Y = random_sample_extractor.get_random_sample_from_class(2)
        X_2.append(X)
        Y_2.append(Y)
    show_slices(X_2, Y_2)
    plt.suptitle("Label=2 slices: liver lesions", size=40)
    plt.show(block=False)


# ## Implement rotation augmentation
#
# Implement the ```get_rnd_rotation()``` function in the class above.
# You can use the functions below to test if the output makes sense.


#Test if we can generate instances of the sample extractor class for all 3D volumes
#sample_extractors=[]
#for i, (vol, seg) in enumerate(zip(tra_vol_batch_1, tra_seg_batch_1)):
#    sample_extractor = SampleExtractor((512, 512),vol, seg, max_rotation=0, rnd_flipping=False)
#    exec "sample_extractor_%s=sample_extractor" % (i)
#    sample_extractors.append(sample_extractor)

test_rotation = False
if test_rotation:
    #Now we enable rotation, and test it again.
    random_sample_extractor = SampleExtractor((512, 512), vol_batch[0], seg_batch[0], max_rotation=10, elastic_deformation=False)
    X_0 = []
    Y_0 = []
    for i in range(5):
        X, Y = random_sample_extractor.get_random_sample_from_class(0)
        X_0.append(X)
        Y_0.append(Y)
    show_slices(X_0, Y_0)
    plt.suptitle("Label=0 slices: no liver", size=40)
    plt.show(block=False)
    X_1 = []
    Y_1 = []
    for i in range(5):
        X, Y = random_sample_extractor.get_random_sample_from_class(1)
        X_1.append(X)
        Y_1.append(Y)
    show_slices(X_1, Y_1)
    plt.suptitle("Label=1 slices: liver without lesions", size=40)
    plt.show(block=False)
    X_2 = []
    Y_2 = []
    for i in range(5):
        X, Y = random_sample_extractor.get_random_sample_from_class(2)
        X_2.append(X)
        Y_2.append(Y)
    show_slices(X_2, Y_2)
    plt.suptitle("Label=2 slices: liver lesions", size=40)
    plt.show(block=False)


'''
Split the data in a training and validation set.
'''

validation_percentage = 0.3
n_validation_images = int(ceil(len(vol_batch)*validation_percentage))

# use the first images as validation
val_vol_batch = vol_batch[0:n_validation_images]
val_seg_batch = seg_batch[0:n_validation_images]

# the rest as training
tra_vol_batch = vol_batch[n_validation_images:]
tra_seg_batch = seg_batch[n_validation_images:]

print("nr of training images: " + str(len(tra_vol_batch)) + "\nnr of validation images: " + str(len(val_vol_batch)))


'''
Set parameters
Set the parameters to suitable values.
Certain paramters have a large influence on the performance of your network, try to find the optimal parameters for
this problem!
'''

learning_rate = 0.01
nr_epochs = 1
nr_iterations_per_epoch = 1
patch_size = (512,512)
batch_size = 1
max_rotation = 10
gaussian_blur = False
elastic_deformation = True
class_balancing = True
nr_validation_samples_per_3D_volume = 1
validation_batch_size = 1
network_name = 'network_LiTS'

ps_x = patch_size[0]
ps_y = patch_size[1]

# ## Compile the theano functions
inputs = T.ftensor4('X')
targets = T.itensor4('Y')
weights = T.ftensor4('W')

# Unet defined with pad='same' so input_size=output_size
# The original Unet is defined without padding to not create border effects in case of tiling a large image.
# Our images are not that large, and taken as a whole, so border effect are no concern here.

liverNetwork = UNet(inputs, input_size=patch_size,
                         depth=5,
                         branching_factor=6,  # 2^6 filters for first level, 2^7 for second, etc.
                         num_input_channels=1,
                         num_classes=2,
                         pad='same')

lesionNetwork = UNet(inputs, input_size=patch_size,
                         depth=5,
                         branching_factor=6,  # 2^6 filters for first level, 2^7 for second, etc.
                         num_input_channels=1,
                         num_classes=2,
                         pad='same')


train_fn, validation_fn = define_updates(network, inputs, targets, weights, learning_rate=0.01, momentum=0.9, l2_lambda=1e-5)

'''
Create the
 - sample extractors
 - batch extractors
 - constant validation set
 - Initialize all monitoring statistics
'''

random.seed(0)

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

# Pick every epoch a random 3D volume for training and for validation
tra_epoch = np.random.choice(len(tra_vol_batch), nr_epochs, replace=True)
val_epoch = np.random.choice(len(val_vol_batch), nr_epochs, replace=True)
# Main training loop
for epoch in range(nr_epochs):
    print('Epoch {}'.format(epoch + 1))
    # Collect training samples
    itra = tra_epoch[epoch]
    sample_ex_train = SampleExtractor(patch_size, tra_vol_batch[itra], tra_seg_batch[itra], labeling='lesion',
                                      max_rotation=max_rotation, elastic_deformation=elastic_deformation)
    batch_ex_train = BatchExtractor(sample_ex_train, batch_size, class_balancing=class_balancing)
    #train_X, train_Y = batch_ex_train.get_random_batch_balanced()
    # Collect validation samples
    ival = val_epoch[epoch]
    sample_ex_val = SampleExtractor(patch_size, val_vol_batch[ival], val_seg_batch[ival], labeling='lesion',
                                    max_rotation=max_rotation, elastic_deformation=elastic_deformation)
    batch_ex_val = BatchExtractor(sample_ex_val, nr_validation_samples_per_3D_volume,
                                  class_balancing=class_balancing)
    #validation_X, validation_Y = batch_ex_val.get_random_batch_balanced()

    tra_loss_lst = []
    val_loss_lst = []
    tra_acc_lst = []
    val_acc_lst = []
    best_val_acc = 0
    # training
    tra_losses = []
    tra_accs = []
    print('training...')
    for b in range(0, nr_iterations_per_epoch),:
        X, Y = batch_ex_train.get_random_batch_balanced()
        #Y = Y / 2
        #Y = Y.astype(np.int16)
        weights_map = np.ndarray(X.shape)
        weights_map.fill(1)
        print 'train X', X.shape, X.dtype, X.min(), X.max(), np.any(np.isnan(X))
        print 'train Y', Y.shape, Y.dtype, Y.min(), Y.max(), np.any(np.isnan(Y))
        print 'train weights_map', weights_map.shape, weights_map.dtype, weights_map.min(), weights_map.max(), np.any(np.isnan(Y))
        loss, l2_loss, accuracy, target_prediction, prediction = \
            train_fn(X.astype(np.float32), Y.astype(np.int32), weights_map.astype(np.float32))
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

    for b in range(0, nr_validation_samples_per_3D_volume, validation_batch_size):
        X, Y = batch_ex_val.get_random_batch_balanced()
        #X = validation_X[b:min(b + validation_batch_size, nr_validation_samples)]
        #Y = validation_Y[b:min(b + validation_batch_size, nr_validation_samples)]
        # print 'test X', X.shape, X.dtype#, X.min(), X.max(), np.any(np.isnan(X))
        # print 'test Y', Y.shape, Y.dtype, Y.min(), Y.max(), np.any(np.isnan(Y))
        loss, l2_loss, accuracy, target_prediction, prediction = \
            validation_fn(X.astype(np.float32), Y.astype(np.int16), weights_map.astype(np.float32))
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
test_dir='./data/Test-Data'

vol_test = sorted(get_file_list(test_dir, 'test-volume')[0])

result_output_folder = os.path.join(test_dir, 'results')
if not (os.path.exists(result_output_folder)):
    os.mkdir(result_output_folder)

# ## Apply the trained network to the images

# indicate the name of the network for this test
network_name = 'network_LiTS'

# initialize the network used in this experiment (this may change)
network = define_network(inputs, input_size=patch_size,
                         depth=5,
                         branching_factor=6,  # 2^6 filters for first level, 2^7 for second, etc.
                         num_input_channels=1,
                         num_classes=2,
                         pad='same')
npz = np.load('./' + network_name + '.npz')  # load stored parameters
lasagne.layers.set_all_param_values(network, npz['params'])  # set parameters
predict_fn = define_predict(network, inputs)

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
        probability = predict_fn(X.astype(np.float32))

        #probability[0] is for no label, probability[1] is for label
        probability_labeled = np.asarray(probability)[:,:,1].reshape(patch_size)

        # binary version by thresholding
        thresholded_image[:, :, slice] = probability_labeled > threshold

    # save output to file
    np.save(vol.replace('volume', 'lesions'), thresholded_image)







