
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib


# function to get a list of file of a given extension, both the absolute path and the filename

def get_file_list(path,ext='',queue=''):
    if ext != '':
        return [os.path.join(path,f) for f in os.listdir(path) if f.startswith(ext)],  \
               [f for f in os.listdir(path) if f.startswith(ext)]
    else:
        return [os.path.join(path,f) for f in os.listdir(path)]

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

def show_slices(vol_slices, seg_slices):
    """ Function to display row of image slices """
    nslice = len(vol_slices)
    fig, axes = plt.subplots(2, nslice)
    for i, slice in enumerate(vol_slices):
        axes[0, i].imshow(slice.T, cmap="gray", origin="lower")
    for i, slice in enumerate(seg_slices):
        axes[1, i].imshow(slice.T, cmap="gray", origin="lower", clim=(0, 2))

def show_slices_x3(vol_slices, seg_slices, pred_slices):
    """ Function to display row of image slices """
    nslice = len(vol_slices)
    fig, axes = plt.subplots(3, nslice)
    for i, slice in enumerate(vol_slices):
        axes[0, i].imshow(slice.T, cmap="gray", origin="lower")
    for i, slice in enumerate(seg_slices):
        axes[1, i].imshow(slice.T, cmap="gray", origin="lower", clim=(0, 2))
    for i, slice in enumerate(pred_slices):
        axes[2, i].imshow(slice.T, cmap="gray", origin="lower", clim=(0, 2))


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
        plt.show(block=False)
        plt.pause(0.5)

def show_preprocessing(nr_tra_volumes, train_batch_dir, batchGenerator, patch_size, out_size, img_center):

    X_0 = []
    Y_0 = []
    for i in range(5):
        i_vol = int(np.random.random() * nr_tra_volumes)
        vol = train_batch_dir + "/volume-{0}.nii".format(i_vol)
        vol_proxy = nib.load(vol)
        vol_array = vol_proxy.get_data()
        seg = train_batch_dir + "/segmentation-{0}.nii".format(i_vol)
        seg_proxy = nib.load(seg)
        seg_array = seg_proxy.get_data()
        slice = int(np.random.random() * vol_array.shape[2])
        vol_slice = vol_array[:, :, slice]
        seg_slice = seg_array[:, :, slice]
        # Execute the following cells several times to see if the results you get make sense
        X, Y = batchGenerator.preprocess_slice(vol_slice, seg_slice, patch_size, out_size, img_center)
        X_0.append(X)
        Y_0.append(Y)
    show_slices(X_0, Y_0)
    plt.suptitle("Preprocessed slices", size=40)
    plt.show(block=False)
    plt.pause(0.5)

def show_segmentation_prediction(batchGenerator, network, vol_list, batch_dir,
                                 patch_size, out_size, img_center):
    nr_test_batches = 5 # batch_size per test
    batch_size = 1
    X_0 = []
    Y_0 = []
    target_preds = []
    val_preds = []
    for i in range(nr_test_batches):
        X_val, Y_val = batchGenerator.get_batch(vol_list, batch_dir, batch_size,
                                                   patch_size, out_size, img_center, group_labels="lesion",
                                                   group_percentages=(0.5, 0.5))
        weights_map = np.ndarray(Y_val.shape)
        weights_map.fill(1)
        _, _, _, _, prediction = \
            network.val_fn(X_val.astype(np.float32), Y_val.astype(np.int32), weights_map.astype(np.float32))
        X_0.append(X_val[0,0,:,:])
        Y_0.append(Y_val[0,0,:,:])
        val_preds.append(prediction.reshape(out_size[0], out_size[1],2)[:,:,1])
    show_slices_x3(X_0, Y_0, val_preds)
    plt.suptitle("Segmentation results for validation images ", size=40)
    plt.savefig('Validation_segmentation.png')
    plt.pause(0.5)