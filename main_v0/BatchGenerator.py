import numpy as np
import random
import nibabel as nib

'''
Batch generator
'''

class BatchGenerator:
    def __init__(self, mask_network, threshold):
        self.mask_network = mask_network
        self.threshold = threshold

    def get_batch(self, vol_list, batch_dir, batch_size, patch_size, out_size, img_center, target_class="liver", group_percentages=(0.5, 0.5)):

        ps_x, ps_y = patch_size
        out_x, out_y = out_size
        nr_volumes = len(vol_list)

        X_batch = np.ndarray((batch_size, 1, ps_y, ps_x))
        Y_batch = np.ndarray((batch_size, 1, out_y, out_x))

        #Get a random volume and a random slice from it, batch_size times
        for b in range(batch_size):

            i_vol = np.random.choice(vol_list, replace=True)#replace=False when using all volumes

            vol = batch_dir + "/volume-{0}.nii".format(i_vol)
            vol_proxy = nib.load(vol)
            vol_array = vol_proxy.get_data()

            seg = batch_dir + "/segmentation-{0}.nii".format(i_vol)
            seg_proxy = nib.load(seg)
            seg_array = seg_proxy.get_data()

            seg_group_labels = self.group_label(seg_array, target_class, group_percentages)

            if b < group_percentages[0]*batch_size:
                label = 0
            else:
                label = 1

            slice = self.select_slice(group_percentages, label) # int(np.random.random() * vol_array.shape[2])

            vol_slice = vol_array[:, :, slice]
            seg_slice = seg_group_labels[:, :, slice]

            # Pre-processing image, i.e. cropping and padding
            X, Y = self.preprocess_slice(vol_slice, seg_slice, patch_size, out_size, img_center)

            #if group_labels == "lesion"
            #X,Y = self.livermask(X,Y,  #get the liver mask from the liver network for the lesion network) (import network we need to get a prediction)
            #Maybe do this before getting a batch (Has to be done for ALL volumes and slices then)

            X_batch[b, 0, :, :] = X
            Y_batch[b, 0, :, :] = Y
        return X_batch, Y_batch

    def group_label(self, seg_array, group_labels, group_percentages):
        # re-label seg_array according to group_labels
        if group_labels == "liver": #(0, (1,2))
            seg_group_labels = (seg_array.astype(np.int32)+1) // 2 #seg_group_labels = [x-1 if x > 1 else x for x in np.nditer(seg_array)]
        elif group_labels == "lesion": #((0,1),2)
            seg_group_labels = seg_array.astype(np.int32) // 2 #[x-1 if x > 0 else x for x in np.nditer(seg_array)]
        else:
            seg_group_labels = seg_array.astype(np.int32)

        lbl_max = np.max(np.max(seg_group_labels, axis=1), axis=0)  # maximum label per slice
        self.lbl_max_0_idx = np.where(lbl_max == 0)[0]  # slice indices of slices with maximum label 0
        self.lbl_max_1_idx = np.where(lbl_max == 1)[0]  # slice indices of slices with maximum label 1

        return seg_group_labels

    def select_slice(self, group_percentages, label):

        #TODO take exactly <group_percentage> slices from each group (not randomly)
        if label == 0 or len(self.lbl_max_1_idx) == 0:
            slice = np.random.choice(self.lbl_max_0_idx,1)
        else:
            slice = np.random.choice(self.lbl_max_1_idx,1)

        return slice[0]

    def preprocess_slice(self, vol_slice, seg_slice, patch_size, out_size, img_center):

        # Mask with ground truth maks
        vol_slice = (seg_slice > 0) * vol_slice

        # Cropping and padding for UNet
        img_size = np.asarray(vol_slice.shape)
        extend = img_size - img_center
        # volume
        patch_array = np.asarray(patch_size)
        crop_begin = np.clip(img_center - patch_array/2, 0, None) # [0, 0]
        crop_end = img_size - np.clip(extend - patch_array/2, 0, None)
        pad_begin = patch_array // 2 - img_center
        pad_end = pad_begin + (crop_end - crop_begin)
        X = np.ndarray(patch_size)
        X[pad_begin[1]:pad_end[1], pad_begin[0]:pad_end[0]] = \
                vol_slice[crop_begin[1]:crop_end[1], crop_begin[0]:crop_end[0]]
        # segmentation
        out_array = np.asarray(out_size)
        crop_begin = np.clip(img_center - out_array / 2, 0, None)
        crop_end = img_size - np.clip(extend - out_array/2, 0, None)
        pad_begin = np.clip(out_array // 2 - img_center, 0, None)
        pad_end = pad_begin + np.clip(crop_end - crop_begin, 0, None)
        Y = np.ndarray(out_size)
        Y[pad_begin[1]:pad_end[1], pad_begin[0]:pad_end[0]] = \
                seg_slice[crop_begin[1]:crop_end[1], crop_begin[0]:crop_end[0]]
        sum_label_outside = np.sum(seg_slice[:crop_begin[1],:]) + np.sum(seg_slice[crop_end[1]:,:]) + \
                               np.sum(seg_slice[:,:crop_begin[0]]) + np.sum(seg_slice[:,crop_end[0]:])
        if sum_label_outside > 10:
            print("Warning: sum labels outside output crop is {}".format(sum_label_outside))

        return X, Y
