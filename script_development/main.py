
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

#from BatchGenerator2 import BatchGenerator
from UNetClass import UNetClass
from tools import get_file_list

import random

#plt.isinteractive()

'''
Data loading
'''

train_batch_dir='../data/Training_Batch'

vol_batch = sorted(get_file_list(train_batch_dir, 'volume')[0])
seg_batch = sorted(get_file_list(train_batch_dir, 'segmentation')[0])

'''
Split the 3D volumes equally in a training and validation set.
'''

nr_volumes = len(vol_batch)
vol_list = range(nr_volumes)
random.shuffle(vol_list)

validation_percentage = 0.5
nr_val_volumes = int(ceil(len(vol_batch)*validation_percentage))
nr_tra_volumes = nr_volumes - nr_val_volumes


# use the first images as validation
tra_vol_list = vol_list[0:nr_tra_volumes]
val_vol_list = vol_list[nr_tra_volumes:]

print("nr of training images: " + str(len(tra_vol_list)) + "\nnr of validation images: " + str(len(val_vol_list)))

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

liverNetwork = UNetClass(inputs, 
					    input_size=(ps_x, ps_y),
					    depth=5,
					    branching_factor=6, # 2^6 filters for first level, 2^7 for second, etc.
					    num_input_channels=1,
					    num_classes=2,
					    pad='same')

#outputs van liverNetwork worden inputs van lesionNetwork
lesionNetwork = UNetClass(inputs, 
					    input_size=(ps_x, ps_y),
                        depth=5,
                        branching_factor=6,  # 2^6 filters for first level, 2^7 for second, etc.
                        num_input_channels=1,
                        num_classes=2,
                        pad='same')



train_fn, validation_fn = liverNetwork.define_updates(inputs, targets, weights)#, learning_rate=0.01, momentum=0.9, l2_lambda=1e-5)

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

'''
TODO:
#trainBatchGenerator (with augmentation)
#valBatchGenerator (without augmentation)
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
    '''
    TODO: for batch in range <amount of training batches(probably 65)>
        #Training
        tra_vol = "../data/volume-{0}.nii".format(tra_vol{batchNo})
        tra_vol_proxy = nib.load(vol)
        tra_vol_array = vol_proxy.get_data()
            
        tra_seg = "../data/segmentation-{0}.nii".format(tra_vol{batchNo})
        tra_seg_proxy = nib.load(seg)
        tra_seg_array = seg_proxy.get_data()
        
        #Validation
        val_vol = "../data/volume-{0}.nii".format(val_vol{batchNo})
        val_vol_proxy = nib.load(vol)
        val_vol_array = vol_proxy.get_data()
            
        val_seg = "../data/segmentation-{0}.nii".format(val_vol{batchNo})
        val_seg_proxy = nib.load(seg)
        val_seg_array = seg_proxy.get_data()
        
        X_train, Y_train = trainBatchGenerator.get_batch(tra_vol_array, tra_seg_array)
        train_fn with X,Y

        X_val, Y_val = valBatchGenerator.get_batch()
        val_fn with (X_val, Y_val)
    '''


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
        params = L.get_all_param_values(liverNetwork)
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







