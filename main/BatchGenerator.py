import numpy as np
import random
import nibabel as nib
from skimage.measure import label
from scipy.ndimage.morphology import binary_closing

'''
Batch generator
'''

class BatchGenerator:
    def __init__(self, tra_list, val_list, mask_network, mask_name, mask_threshold, train_batch_dir, target_class,
                read_slices, slice_files, nr_slices_per_volume, patch_size, out_size, img_center, group_percentages):
        self.mask_network = mask_network
        self.mask_name = mask_name
        self.mask_threshold = mask_threshold
        self.train_batch_dir = train_batch_dir
        self.patch_size = patch_size
        self.out_size = out_size
        self.img_center = img_center

        # If slices are already saved to a np array, read them from those fles (train, val and masks)
        if read_slices:
            print ('Reading slices...')
            self.vol_tra_slices = np.load(slice_files[0])
            print ('{} vol_tra_slices min {} max {} '.format(self.vol_tra_slices.shape[0], np.min(self.vol_tra_slices), np.max(self.vol_tra_slices)))
            self.seg_tra_slices = np.load(slice_files[1])
            print('{} seg_tra_slices min {} max {} '.format(self.seg_tra_slices.shape[0],np.min(self.seg_tra_slices), np.max(self.seg_tra_slices)))
            self.msk_tra_slices = np.load(slice_files[2])
            print('{} msk_tra_slices min {} max {} '.format(self.msk_tra_slices.shape[0], np.min(self.msk_tra_slices), np.max(self.msk_tra_slices)))
            self.n_slices = self.vol_tra_slices.shape[0]
            self.vol_val_slices = np.load(slice_files[3])
            print('{} vol_val_slices min {} max {} '.format(self.vol_val_slices.shape[0],np.min(self.vol_val_slices), np.max(self.vol_val_slices)))
            self.seg_val_slices = np.load(slice_files[4])
            print('{} seg_val_slices min {} max {} '.format(self.seg_val_slices.shape[0],np.min(self.seg_val_slices), np.max(self.seg_val_slices)))
            self.msk_val_slices = np.load(slice_files[5])
            print('{} msk_val_slices min {} max {} '.format(self.msk_val_slices.shape[0], np.min(self.msk_val_slices), np.max(self.msk_val_slices)))
            self.n_val_slices = self.vol_val_slices.shape[0]
            print('Done reading slices.')
            
        # Else if slices are to be converted to np arrays, pick 'nr_slices_per_volume' amount of slices per volume for both train, val and masks
        else:
            print('Collecting slices...')
            self.n_tra_slices = nr_slices_per_volume * len(tra_list)
            self.vol_tra_slices, self.seg_tra_slices, self.msk_tra_slices = self.collect_training_slices(tra_list, self.n_tra_slices,
                                                        nr_slices_per_volume, target_class, group_percentages, train=True)
            np.save('vol_tra_slices', self.vol_tra_slices)
            np.save('seg_tra_slices', self.seg_tra_slices)
            np.save('msk_tra_slices', self.msk_tra_slices)
            self.n_val_slices = nr_slices_per_volume * len(val_list)
            self.vol_val_slices, self.seg_val_slices, self.msk_val_slices = self.collect_training_slices(val_list, self.n_val_slices,
                                                        nr_slices_per_volume, target_class, group_percentages, train=False)
            np.save('vol_val_slices', self.vol_val_slices)
            np.save('seg_val_slices', self.seg_val_slices)
            np.save('msk_val_slices', self.msk_val_slices)
            print('Done collecting slices.')

    # Load nifti files with volumes and segmentations, plus make corresponding masks, and save to np arrays
    def collect_training_slices(self, vol_list, n_slices, nr_slices_per_volume, target_class, group_percentages, train=True):
        vol_slices = np.zeros((n_slices, 512, 512))
        seg_slices = np.zeros((n_slices, 512, 512))
        msk_slices = np.zeros((n_slices, 512, 512))
        i_slice = 0
        for i, i_vol in enumerate(vol_list):
            print("get {} slices from {}-th volume #{}".format(nr_slices_per_volume, i, i_vol))

            # Reading in of the volume data (per volume)
            vol = self.train_batch_dir+"/volume-{0}.nii".format(i_vol)
            vol_proxy = nib.load(vol)
            vol_array = vol_proxy.get_data()

            # Reading corresponding labels (per volume)
            seg = self.train_batch_dir+"/segmentation-{0}.nii".format(i_vol)
            seg_proxy = nib.load(seg)
            seg_array = seg_proxy.get_data()
            name = vol.split('-')[1].split('.')[0]

            # Apply normalization on the whole volume
            vol_array = np.clip(vol_array, -200, 300)
            vol_array = (vol_array - vol_array.mean()) / vol_array.std()

            # re-label seg_array according to group_labels
            seg_label,lbl_max_0_idx,lbl_max_1_idx  = self.group_label(seg_array, target_class)

            # Make perumations of length of slices to prevent double visiting of slices
            rand_all = np.random.permutation(seg_label.shape[2])
            rand_0 = lbl_max_0_idx
            np.random.shuffle(rand_0)
            rand_1 = lbl_max_1_idx
            np.random.shuffle(rand_1)
            for s in range(nr_slices_per_volume):
                if(train == True):
                    label = 1.0 * (np.random.random() > group_percentages[0])
                    if ((label == 0 or len(rand_1)==0) and not(len(rand_0)==0)):
                        slice_nr = rand_0[0]
                        rand_0 = np.delete(rand_0, [0], None)
                    else:
                        slice_nr = rand_1[0]
                        rand_1 = np.delete(rand_1, [0], None)
                else:
                    slice_nr = rand_all[s]
                
                # If lesion netwerk, then apply mask
                X_mask = np.zeros((seg_array.shape[0],seg_array.shape[1]))
                if target_class == 'lesion':
                    
                    if self.mask_name == 'liver_network':
                        #Reshape for correct input size for Unet
                        X_mask = vol_array[:, :, slice_nr].reshape(1,1,vol_array.shape[0],vol_array.shape[1])
                        
                        # Predict liver mask
                        X_mask = self.mask_network.predict_fn((X_mask).astype(np.float32))
                        
                        # Reshape image back to 2 dimensions
                        X_mask = X_mask[0].reshape(1, 1, self.out_size[0], self.out_size[1], 2)[:, :, :, :, 1]
                        
                        # Threshold to binary output
                        X_mask = (X_mask > self.mask_threshold).astype(np.int32)

                        # Find the biggest connected component in the liver segmentation
                        X_mask[0, 0, :, :] = self.get_biggest_component(X_mask[0,0,:,:])
                        
                        # Pad back to original image size
                        #X_mask = self.pad(X_mask, (512,512))
                        X_mask = self.pad(X_mask, (512,512), self.img_center)
                        X_mask = np.squeeze(X_mask)
                        
                    elif self.mask_name == 'ground_truth':
                        X_mask = (seg_array[:, :, slice_nr]+1)//2 # temporarily use ground truth mask for lesion network

                vol_slices[i_slice, :, :] = vol_array[:, :, slice_nr]
                seg_slices[i_slice, :, :] = seg_label[:, :, slice_nr]
                msk_slices[i_slice, :, :] = X_mask
                i_slice += 1

        return vol_slices, seg_slices, msk_slices

    # Generator function to get batches
    def get_batch(self, batch_size, train=True):

        if (train):
            n_slices = self.vol_tra_slices.shape[0]
        else:
            n_slices = self.vol_val_slices.shape[0]

        perm = np.random.permutation(n_slices)

        X_batch = np.zeros((batch_size, 1, 512, 512))
        Y_batch = np.zeros((batch_size, 1, 512, 512))
        M_batch = np.zeros((batch_size, 1, 512, 512))

        slice_nr = 0
        while (slice_nr + batch_size < n_slices):
            
            for i in range(batch_size):
                if (train):
                    X_batch[i, 0, :, :] = self.vol_tra_slices[perm[slice_nr], :, :]
                    Y_batch[i, 0, :, :] = self.seg_tra_slices[perm[slice_nr], :, :]
                    M_batch[i, 0, :, :] = self.msk_tra_slices[perm[slice_nr], :, :]
                else:
                    X_batch[i, 0, :, :] = self.vol_val_slices[perm[slice_nr], :, :]
                    Y_batch[i, 0, :, :] = self.seg_val_slices[perm[slice_nr], :, :]
                    M_batch[i, 0, :, :] = self.msk_val_slices[perm[slice_nr], :, :]
                    
                slice_nr += 1
                
            yield (X_batch, Y_batch, M_batch)
        print("No more batches, epoch is done!")

    # Determine which slices contain the positive class (liver or lesion) and which don't
    def group_label(self, seg_array, group_labels):
        # re-label seg_array according to group_labels
        if group_labels == "liver": #(0, (1,2))
            lbl_max = np.max(np.max(seg_array, axis=1), axis=0)  # maximum label per slice
            lbl_max_0_idx = np.where(lbl_max == 0)[0]  # slice indices of slices with maximum label 0
            lbl_max_1_idx = np.where(lbl_max > 0)[0]  # slice indices of slices with maximum label 1
            seg_group_labels = (seg_array.astype(
                np.int32) + 1) // 2  # seg_group_labels = [x-1 if x > 1 else x for x in np.nditer(seg_array)]
        elif group_labels == "lesion": #((0,1),2)
            lbl_max = np.max(np.max(seg_array, axis=1), axis=0)  # maximum label per slice
            lbl_max_0_idx = np.where(lbl_max < 2)[0]   # slice indices of slices with maximum label 0
            lbl_max_1_idx = np.where(lbl_max == 2)[0]  # slice indices of slices with maximum label 1
            seg_group_labels = seg_array.astype(np.int32) // 2  # [x-1 if x > 0 else x for x in np.nditer(seg_array)]
        else:
            lbl_max = np.max(np.max(seg_array, axis=1), axis=0)  # maximum label per slice
            lbl_max_0_idx = np.where(lbl_max < 2)[0]  # slice indices of slices with maximum label 0
            lbl_max_1_idx = np.where(lbl_max == 2)[0]  # slice indices of slices with maximum label 1
            seg_group_labels = seg_array.astype(np.int32)

        return seg_group_labels, lbl_max_0_idx, lbl_max_1_idx 

    # Function to pad an image
    def pad(self, batch, target_size, pad_value = 0):

        # Create array to place image in
        padded = np.ones((batch.shape[0], batch.shape[1], target_size[0], target_size[1])) * pad_value
	
        # Determine start (x, y) of image in padded array
        offsetX = int((target_size[0] - batch.shape[2])/2)
        offsetY = int((target_size[1] - batch.shape[3])/2)
	
        # Place image at target location
        padded[:, :, offsetX:target_size[0]-offsetX, offsetY:target_size[1]-offsetY] = batch[:, :, :, :]
	
        return padded

    # Function to crop an image	
    def crop(self, batch, target_size):
    
        # Create array to place image in
        cropped = np.zeros((batch.shape[0], batch.shape[1], target_size[0], target_size[1]))
	
	# Determine start (x, y) of image in padded array
        offsetX = int((batch.shape[2] - target_size[0])/2)
        offsetY = int((batch.shape[3] - target_size[1])/2)
	
	# Crop image
        cropped[:, :, :, :] = batch[:, :, offsetX:batch.shape[2]-offsetX, offsetY:batch.shape[3]-offsetY]

        # Print how many liver/lesion is outside of crop
        if((np.sum(batch) - np.sum(cropped)) > 10):
            print("Warning: sum labels outside output crop is {}".format(np.sum(batch) - np.sum(cropped)))

        return cropped

    # Keep the biggest connected component and apply binary closing
    def get_biggest_component(self, pred):

        labelled = label(pred, connectivity=2)
        largest_connected_val = 0
        largest_connected_mean = 0

        for val in range(1, labelled.max() +1):
             # We can use mean since divisor is always image area and labelled value is always 1
             connected_mean = (labelled == val).astype(np.int32).mean()

             # Make sure great component is selected
             if (connected_mean > largest_connected_mean):
                 largest_connected_val = val
                 largest_connected_mean = connected_mean

        # Select only largest component from image
        pred[labelled != largest_connected_val] = 0

        # Perform binary closing
        pred = binary_closing(pred).astype(int)

        return pred

'''

    # Get a batch of training imgs (with corresponding labels and masks)
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

    # Get a batch of validation imgs (with corresponding labels and masks)
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

'''
