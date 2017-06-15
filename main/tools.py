
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib

from BatchGenerator import BatchGenerator
from BatchAugmenter import BatchAugmenter

'''
Useful functions to use for plotting and loading
'''

# Function to get a list of file of a given extension, both the absolute path and the filename
def get_file_list(path,ext='',queue=''):
    if ext != '':
        return [os.path.join(path,f) for f in os.listdir(path) if f.startswith(ext)],  \
               [f for f in os.listdir(path) if f.startswith(ext)]
    else:
        return [os.path.join(path,f) for f in os.listdir(path)]

# Plot image, label and mask
def show_image(idx, imgs, msks, lbls):
    img = np.asarray(Image.open(imgs[idx]))
    print
    img.shape
    msk = np.asarray(Image.open(msks[idx]))
    lbl = np.asarray(Image.open(lbls[idx]))
    plt.subplot(1, 3, 1)
    plt.imshow(img);
    plt.title('RGB image {}'.format(idx + 1))
    plt.subplot(1, 3, 2)
    plt.imshow(msk, cmap='gray');
    plt.title('Mask {}'.format(idx + 1))
    plt.subplot(1, 3, 3)
    plt.imshow(lbl, cmap='gray');
    plt.title('Manual annotation {}'.format(idx + 1))
    plt.show()

# Function to display row of image slices
def show_slices(vol_slices, seg_slices):
    nslice = len(vol_slices)
    fig, axes = plt.subplots(2, nslice)
    for i, slice in enumerate(vol_slices):
        axes[0, i].imshow(slice.T, cmap="gray", origin="lower")
    for i, slice in enumerate(seg_slices):
        axes[1, i].imshow(slice.T, cmap="gray", origin="lower", clim=(0, 2))

# Function to display row of image slices
def show_slices_x4(vol_slices, seg_slices, label_slices, pred_slices):
    nslice = len(vol_slices)
    fig, axes = plt.subplots(4, nslice)
    for i, slice in enumerate(vol_slices):
        axes[0, i].imshow(slice.T, cmap="gray", origin="lower")
    for i, slice in enumerate(seg_slices):
        axes[1, i].imshow(slice.T, cmap="gray", origin="lower", clim=(0, 2))
    for i, slice in enumerate(label_slices):
        axes[2, i].imshow(slice.T, cmap="gray", origin="lower", clim=(0, 2))
    for i, slice in enumerate(pred_slices):
        axes[3, i].imshow(slice.T, cmap="gray", origin="lower", clim=(0, 2))

# Determines the output size of Unet given the input and depth
def output_size_for_input(in_size, depth):
    in_size = np.array(in_size)
    in_size -= 4
    for _ in range(depth-1):
        in_size = in_size//2
        in_size -= 4
    for _ in range(depth-1):
        in_size = in_size*2
        in_size -= 4
    return in_size

# Plot the center slices of a volume in every plane
def show_volumes(vol_batch, seg_batch):
    for i, (vol, seg) in enumerate(zip(vol_batch, seg_batch)):
        vol_proxy = nib.load(vol)
        vol_array = vol_proxy.get_data()
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
        #plt.show(block=False)
        #plt.pause(0.5)
        plt.savefig('volume_'+i+'.png')

# Plot the effect of padding and cropping on images and labels
def show_preprocessing(batchGenerator, augmenter, aug_params, \
                       patch_size, out_size, img_center, target_class):

    X_0 = []
    Y_0 = []
    for i,batch in enumerate(batchGenerator.get_batch(batch_size = 1, train=True)):
        # Generate batch
        X_tra, Y_tra, M_tra = batch

        # ROI METHOD WERE WE PUT EVERY PIXEL OUTSIDE OF LIVER TO ZERO
        if(target_class == 'lesion' and not(mask_name == None)):
            X_tra[np.where(M_tra == 0)] = np.min(X_tra)
            
        # Augment data batch
        X_tra, Y_tra, M_tra = augmenter.getAugmentation(X_tra, Y_tra, M_tra, aug_params)

        # Pad X and crop Y for UNet, note that array dimensions change here!
        X_tra = batchGenerator.pad(X_tra, patch_size, pad_value=np.min(X_tra))
        Y_tra = batchGenerator.crop(Y_tra, out_size)
        M_tra = batchGenerator.crop(M_tra, out_size)
        
        X_0.append(X_tra[0, 0, :, :])
        Y_0.append(Y_tra[0, 0, :, :])
        print('Show preprocessing X min {:.2f} max {:.2f}, Y min {:.2f} max {:.2f}'.format(
            np.min(X_tra), np.max(X_tra), np.min(Y_tra), np.max(Y_tra)))

        # Because get_batch can generate a lot more batches
        if(i>=4):
            break

    show_slices(X_0, Y_0)
    plt.suptitle("Preprocessed slices", size=40)
    #plt.show(block=False)
    #plt.pause(0.5)
    plt.savefig('Preprocess.png')

# Plot segmentation results
def show_segmentation_prediction(trainer, network, mask_threshold, val_list, train_batch_dir,
                                 patch_size, out_size, img_center, target_class,
                                 read_slices, slice_files, nr_slices_per_volume, weight_balance, mask_name, mask_network):

    nr_test_batches = 5 # batch_size per test

    # Define batch generator
    batchGenerator = BatchGenerator(val_list, val_list, mask_network, mask_name, mask_threshold,
                                    train_batch_dir, target_class, read_slices, slice_files,
                                    nr_slices_per_volume, patch_size, out_size, img_center,
                                    group_percentages=(0.5, 0.5)) 

    X_0 = []
    Y_0 = []
    target_preds = []
    val_preds = []
    val_labels = []
    batch_size = 1
    for i,batch in enumerate(batchGenerator.get_batch(batch_size = 1, train=True)):
        X_val, Y_val, M_val = batch

        # ROI METHOD WERE WE PUT EVERY PIXEL OUTSIDE OF LIVER TO ZERO
        if(target_class == 'lesion' and not(mask_name == None)):
            X_val[np.where(M_val == 0)] = np.min(X_val)
            
        # Pad X and crop Y for UNet, note that array dimensions change here!
        X_val = batchGenerator.pad(X_val, patch_size, pad_value=np.min(X_val))
        Y_val = batchGenerator.crop(Y_val, out_size)
        M_val = batchGenerator.crop(M_val, out_size)

        prediction, loss, accuracy = trainer.validate_batch(network, X_val, Y_val, M_val, weight_balance, target_class)
        prediction = prediction.reshape(batch_size, 1, out_size[0], out_size[1], 2)[:, :, :, :, 1]

        pred_binary = (prediction[0,0,:,:] > mask_threshold).astype(np.int32)
        # Find the biggest connected component in the liver segmentation
        if(target_class == 'liver'):
            pred_binary = batchGenerator.get_biggest_component(pred_binary)

        X_0.append(X_val[0,0,:,:])
        Y_0.append(Y_val[0,0,:,:])
        val_preds.append(prediction[0,0,:,:])
        val_labels.append(pred_binary)

        if(i>=nr_test_batches):
            break;        
        
    show_slices_x4(X_0, Y_0, val_labels, val_preds)
    plt.suptitle("Segmentation results for validation images ", size=40)
    plt.savefig('Validation_segmentation.png')
    #plt.pause(0.5)
