import numpy as np
import random
import nibabel as nib
from skimage.measure import label
from scipy.ndimage.morphology import binary_closing

'''
Tester
'''

class Tester:
    def __init__(self, patch_size, out_size, liver_network, liver_threshold, lesion_network, lesion_threshold):

        self.patch_size = patch_size
        self.out_size = out_size
        self.liver_network = liver_network
        self.liver_threshold = liver_threshold
        self.lesion_network = lesion_network
        self.lesion_threshold = lesion_threshold


    # Use given network to perform testing on the test images
    def perform_test(self, test_list, test_batch_dir):

        # Make sure that the output folder exists
        result_output_folder = os.path.join(test_batch_dir, 'results')
        if not (os.path.exists(result_output_folder)):
            os.mkdir(result_output_folder)

        # Define Image size
        img_x, img_y = (512, 512)

        # Iterate over every test volume
        for i_vol in test_list:

            test_batch = collect_testing_slices(i_vol, test_batch_dir)
            classification = np.zeros((test_batch[1], test_batch[2], test_batch[0]))

            # Iterate over each slice in volume
            for j,test_slice in enumerate(test_batch):
                test_slice = test_slice.reshape(1, 1, test_slice[0], test_slice[1])
                
                # Pad test images
                test_slice = self.pad(test_slice, self.patch_size, pad_value = np.min(test_slice))

                # Apply liver segmentation network
                liver_seg_mask = self.liver_network.predict_fn(test_slice.astype(np.float32))
                liver_prediction = liver_seg_mask[0].reshape(1, 1, self.out_size[0], self.out_size[1], 2)[:, :, :, :, 1]
                
                # Turn heatmap into binary classification and pad again
                liver_seg_mask = (liver_prediction > self.liver_threshold).astype(np.int32)
                liver_seg_mask = self.pad(liver_seg_mask, self.patch_size)

                # Find the biggest connected component in the liver segmentation
                liver_seg_mask[0, 0, :, :] = self.get_biggest_component(liver_seg_mask[0,0,:,:])

                # Apply segmentation mask as ROI on test images
                test_slice[np.where(liver_seg_mask == 0)] = np.min(test_slice)
                
                # Apply lesion detection network
                lesion_seg_mask = self.lesion_network.predict_fn(test_slice.astype(np.float32))
                lesion_prediction = lesion_seg_mask[0].reshape(1, 1, self.out_size[0], self.out_size[1], 2)[:, :, :, :, 1]
                
                # Turn heatmap into binary classification and pad again
                lesion_seg_mask = (lesion_prediction > self.lesion_threshold).astype(np.int32)
                lesion_seg_mask = self.pad(lesion_seg_mask, (512,512))

                # Match format (lesion has value 2)
                lesion_seg_mask = lesion_seg_mask * 2

                # Squeeze then tranpose
                lesion_seg_mask = np.squeeze(lesion_seg_mask)
                
                # Then save into classification array
                classification[:,:, j] = lesion_seg_mask

            # Turn image into .nii file
            nii_classification = nib.Nifti1Image(classification, affine=affine_shape)

            # Save output to file
            nib.save(nii_classification, os.path.join(test_batch_dir, "results/test-segmentation-{}.nii".format(i_vol)))        


    # Load all test files from all test volumes in an npz file
    def collect_testing_slices(self, i_vol, test_batch_dir):
        test_slices = []
       
        print("Get slices from volume #{}".format(i_vol))

        # Reading in of the volume data (per volume)
        vol = test_batch_dir+"/test-volume-{0}.nii".format(i_vol)
        vol_proxy = nib.load(vol)
        vol_array = vol_proxy.get_data()

        # Apply normalization on the whole volume
        vol_array = np.clip(vol_array, -200, 300)
        vol_array = (vol_array - vol_array.mean()) / vol_array.std()

        # Appending every slice from the volume to test_slices
        for s in range(vol_array.shape[2]):
            test_slices.append(vol_array[:, :, s])

        # Return as np array
        test_slices = np.asarray(test_slices)

        return test_slices


    def pad(self, batch, target_size, pad_value = 0):

        # Create array to place image in
        padded = np.ones((batch.shape[0], batch.shape[1], target_size[0], target_size[1])) * pad_value
	
        # Determine start (x, y) of image in padded array
        offsetX = int((target_size[0] - batch.shape[2])/2)
        offsetY = int((target_size[1] - batch.shape[3])/2)
	
        # Place image at target location
        padded[:, :, offsetX:target_size[0]-offsetX, offsetY:target_size[1]-offsetY] = batch[:, :, :, :]
	
        return padded


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
