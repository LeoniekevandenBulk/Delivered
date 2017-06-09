import numpy as np
import random
import nibabel as nib

'''
Batch generator
'''

class BatchGenerator:
    def __init__(self, mask_network, threshold, tra_list, val_list, train_batch_dir, target_class,
                read_slices, slice_files, nr_slices_per_volume, group_percentages):
        self.mask_network = mask_network
        self.threshold = threshold
        self.train_batch_dir = train_batch_dir

        if read_slices:
            print ('Reading slices...')
            self.vol_tra_slices = np.load(slice_files[0]) # Why is this required ?!
            print ('{} vol_tra_slices min {} max {} '.format(self.vol_tra_slices.shape[0], np.min(self.vol_tra_slices), np.max(self.vol_tra_slices)))
            self.seg_tra_slices = np.load(slice_files[1])
            print('{} seg_tra_slices min {} max {} '.format(self.seg_tra_slices.shape[0],np.min(self.seg_tra_slices), np.max(self.seg_tra_slices)))
            self.msk_tra_slices = np.load(slice_files[2])
            print('{} msk_tra_slices min {} max {} '.format(self.seg_tra_slices.shape[0], np.min(self.seg_tra_slices), np.max(self.seg_tra_slices)))
            self.n_slices = self.vol_tra_slices.shape[0]
            self.vol_val_slices = np.load(slice_files[3]) # Why is this required ?!
            print('{} vol_val_slices min {} max {} '.format(self.vol_val_slices.shape[0],np.min(self.vol_val_slices), np.max(self.vol_val_slices)))
            self.seg_val_slices = np.load(slice_files[4])
            print('{} seg_val_slices min {} max {} '.format(self.seg_val_slices.shape[0],np.min(self.seg_val_slices), np.max(self.seg_val_slices)))
            self.seg_val_slices = np.load(slice_files[4])
            print('{} msk_val_slices min {} max {} '.format(self.seg_val_slices.shape[0], np.min(self.seg_val_slices), np.max(self.seg_val_slices)))
            self.n_val_slices = self.vol_val_slices.shape[0]
            print('Done reading slices.')
        else:
            print('Collecting slices...')
            self.n_tra_slices = nr_slices_per_volume * len(tra_list)
            self.vol_tra_slices, self.seg_tra_slices, self.msk_tra_slices = self.collect_training_slices(tra_list, self.n_tra_slices,
                                                        nr_slices_per_volume, target_class, group_percentages)
            np.save('vol_tra_slices', self.vol_tra_slices)
            np.save('seg_tra_slices', self.seg_tra_slices)
            np.save('msk_tra_slices', self.msk_tra_slices)
            self.n_val_slices = nr_slices_per_volume * len(val_list)
            self.vol_val_slices, self.seg_val_slices, self.msk_val_slices = self.collect_training_slices(val_list, self.n_val_slices,
                                                        nr_slices_per_volume, target_class, group_percentages)
            np.save('vol_val_slices', self.vol_val_slices)
            np.save('seg_val_slices', self.seg_val_slices)
            np.save('msk_val_slices', self.msk_val_slices)
            print('Done collecting slices.')


    def collect_training_slices(self, vol_list, n_slices, nr_slices_per_volume, target_class, group_percentages):
        vol_slices = np.zeros((n_slices, 512, 512))
        seg_slices = np.zeros((n_slices, 512, 512))
        msk_slices = np.zeros((n_slices, 512, 512))
        i_slice = 0
        for i, i_vol in enumerate(vol_list):
            print("get {} slices from {}-th volume #{}".format(nr_slices_per_volume, i, i_vol))
            vol = self.train_batch_dir+"/volume-{0}.nii".format(i_vol)
            vol_proxy = nib.load(vol)
            vol_array = vol_proxy.get_data()
            seg = self.train_batch_dir+"/segmentation-{0}.nii".format(i_vol)
            seg_proxy = nib.load(seg)
            seg_array = seg_proxy.get_data()
            name = vol.split('-')[1].split('.')[0]

            # re-label seg_array according to group_labels
            seg_label = self.group_label(seg_array, target_class)

            for s in range(nr_slices_per_volume):
                label = 1.0 * (np.random.random() > group_percentages[0])
                slice = self.select_slice(label)
                #slice = np.int(np.random.random() * vol_array.shape[2])

                # If lesion netwerk, then apply mask
                X_mask = np.zeros(seg_array.shape)
                if target_class == 'lesion':
                    mask = 'ground_truth'
                    if mask == 'liver_network':
                        X_mask = self.mask_network.predict(vol_array[:, :, slice])
                    elif mask == 'ground_truth':
                        X_mask = (seg_array[:, :, slice]+1)//2 # temporarily use ground truth mask for lesion network
                    #vol_array[:, :, slice] = (X_mask > 0.5).astype(np.int32) * vol_array[:, :, slice]

                vol_slices[i_slice, :, :] = vol_array[:, :, slice]
                seg_slices[i_slice, :, :] = seg_label[:, :, slice]
                msk_slices[i_slice, :, :] = X_mask
                i_slice += 1

        # Clip, then normalize to [0 1]
        vol_slices = np.clip(vol_slices, -200, 300)
        vol_slices = (vol_slices + 200) / 500

        # Per slice apply zero mean std 1 equalization. CAUSES NaN values in vol_slices !!
        #vol_slices = np.clip((vol_slices - np.mean(vol_slices,axis=0)) / np.std(vol_slices,axis=0), -3, 3)

        return vol_slices, seg_slices, msk_slices

    def get_train_batch(self, batch_size):

        img_x, img_y = (512, 512)

        X_batch = np.ndarray((batch_size, 1, img_y, img_x))
        Y_batch = np.ndarray((batch_size, 1, img_y, img_x))
        M_batch = np.ndarray((batch_size, 1, img_y, img_x))

        n_slices = self.vol_tra_slices.shape[0]

        #Get a random volume and a random slice from it, batch_size times
        for b in range(batch_size):

            slice = np.int(np.random.random() * n_slices)

            X = self.vol_tra_slices[slice, :, :]
            Y = self.seg_tra_slices[slice, :, :]
            M = self.msk_tra_slices[slice, :, :]

            X_batch[b, 0, :, :] = X
            Y_batch[b, 0, :, :] = Y
            M_batch[b, 0, :, :] = M

        return X_batch, Y_batch, M_batch # temporarily pass on ground truth mask for lesion network

    def get_val_batch(self, batch_size):

        img_x, img_y = (512, 512)

        X_batch = np.ndarray((batch_size, 1, img_y, img_x))
        Y_batch = np.ndarray((batch_size, 1, img_y, img_x))
        M_batch = np.ndarray((batch_size, 1, img_y, img_x))

        n_slices = self.vol_val_slices.shape[0]

        #Get a random volume and a random slice from it, batch_size times
        for b in range(batch_size):

            slice = np.int(np.random.random() * n_slices)

            X = self.vol_val_slices[slice, :, :]
            Y = self.seg_val_slices[slice, :, :]
            M = self.msk_tra_slices[slice, :, :]

            X_batch[b, 0, :, :] = X
            Y_batch[b, 0, :, :] = Y
            M_batch[b, 0, :, :] = M

        return X_batch, Y_batch, M_batch # temporarily pass on ground truth mask for lesion network

    def group_label(self, seg_array, group_labels):
        # re-label seg_array according to group_labels
        if group_labels == "liver": #(0, (1,2))
            lbl_max = np.max(np.max(seg_array, axis=1), axis=0)  # maximum label per slice
            self.lbl_max_0_idx = np.where(lbl_max == 0)[0]  # slice indices of slices with maximum label 0
            self.lbl_max_1_idx = np.where(lbl_max > 0)[0]  # slice indices of slices with maximum label 1
            seg_group_labels = (seg_array.astype(
                np.int32) + 1) // 2  # seg_group_labels = [x-1 if x > 1 else x for x in np.nditer(seg_array)]
        elif group_labels == "lesion": #((0,1),2)
            lbl_max = np.max(np.max(seg_array, axis=1), axis=0)  # maximum label per slice
            self.lbl_max_0_idx = np.where(lbl_max == 1)[0]  # slice indices of slices with maximum label 0
            self.lbl_max_1_idx = np.where(lbl_max == 2)[0]  # slice indices of slices with maximum label 1
            seg_group_labels = seg_array.astype(np.int32) // 2  # [x-1 if x > 0 else x for x in np.nditer(seg_array)]
        else:
            lbl_max = np.max(np.max(seg_array, axis=1), axis=0)  # maximum label per slice
            self.lbl_max_0_idx = np.where(lbl_max < 2)[0]  # slice indices of slices with maximum label 0
            self.lbl_max_1_idx = np.where(lbl_max == 2)[0]  # slice indices of slices with maximum label 1
            seg_group_labels = seg_array.astype(np.int32)

        return seg_group_labels

    def select_slice(self, label):

        #TODO take exactly <group_percentage> slices from each group (not randomly)
        if label == 0 or len(self.lbl_max_1_idx) == 0:
            slice = np.random.choice(self.lbl_max_0_idx,1)
        else:
            slice = np.random.choice(self.lbl_max_1_idx,1)

        return slice[0]

    def pad_and_crop(self, Ximg, Yimg, Mimg, patch_size, out_size, img_center):

        # Cropping and padding for UNet
        img_size = np.asarray((Ximg.shape[2], Ximg.shape[3]))
        extend = img_size - img_center
        # pad X
        patch_array = np.asarray(patch_size)
        crop_begin = np.clip(img_center - patch_array/2, 0, None) # [0, 0]
        crop_end = img_size - np.clip(extend - patch_array/2, 0, None)
        pad_begin = patch_array // 2 - img_center
        pad_end = pad_begin + (crop_end - crop_begin)
        X = np.zeros((Ximg.shape[0], Ximg.shape[1], patch_size[0], patch_size[1]))
        X[:, :, pad_begin[1]:pad_end[1], pad_begin[0]:pad_end[0]] = \
                Ximg[:, :, crop_begin[1]:crop_end[1], crop_begin[0]:crop_end[0]]
        # crop Y
        out_array = np.asarray(out_size)
        crop_begin = np.clip(img_center - out_array / 2, 0, None)
        crop_end = img_size - np.clip(extend - out_array/2, 0, None)
        pad_begin = np.clip(out_array // 2 - img_center, 0, None)
        pad_end = pad_begin + np.clip(crop_end - crop_begin, 0, None)
        Y = np.zeros((Yimg.shape[0], Yimg.shape[1], out_size[0], out_size[1]))
        Y[:, :, pad_begin[1]:pad_end[1], pad_begin[0]:pad_end[0]] = \
                Yimg[:, :, crop_begin[1]:crop_end[1], crop_begin[0]:crop_end[0]]
        M = np.zeros((Mimg.shape[0], Mimg.shape[1], out_size[0], out_size[1]))
        M[:, :, pad_begin[1]:pad_end[1], pad_begin[0]:pad_end[0]] = \
                Mimg[:, :, crop_begin[1]:crop_end[1], crop_begin[0]:crop_end[0]]
        sum_label_outside = np.sum(Yimg[:,:,:crop_begin[1],:]) + np.sum(Yimg[:,:,crop_end[1]:,:]) + \
                               np.sum(Yimg[:,:,:,:crop_begin[0]]) + np.sum(Yimg[:,:,:,crop_end[0]:])
        if sum_label_outside > 10:
            print("Warning: sum labels outside output crop is {}".format(sum_label_outside))

        return X, Y, M

    def pad(self, Ximg, patch_size, img_center):

        # Cropping and padding for UNet
        img_size = np.asarray((Ximg.shape[2], Ximg.shape[3]))
        extend = img_size - img_center

        # pad X
        patch_array = np.asarray(patch_size)
        crop_begin = np.clip(img_center - patch_array / 2, 0, None)  # [0, 0]
        crop_end = img_size - np.clip(extend - patch_array / 2, 0, None)
        pad_begin = patch_array // 2 - img_center
        pad_end = pad_begin + (crop_end - crop_begin)
        X = np.ndarray((Ximg.shape[0], Ximg.shape[1], patch_size[0], patch_size[1]))
        X[:, :, pad_begin[1]:pad_end[1], pad_begin[0]:pad_end[0]] = \
            Ximg[:, :, crop_begin[1]:crop_end[1], crop_begin[0]:crop_end[0]]

        return X
