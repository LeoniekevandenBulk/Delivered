from BatchAugmenter import *
import numpy as np
from PIL import Image
from tools import get_file_list
import nibabel as nib

train_batch_dir='../data/Training_Batch'

vol_batch = sorted(get_file_list(train_batch_dir, 'volume')[0])
seg_batch = sorted(get_file_list(train_batch_dir, 'segmentation')[0])

'''
img1 = Image.open("liver.jpg")
img2 = Image.open("liver.jpg")
img_batch = [np.array(img1)[:,:,1], np.array(img2)[:,:,1]]
img_labels = [np.array(img1)[:,:,1], np.array(img2)[:,:,1]]'''
myAugmenter = BatchAugmenter()
i_vol =1
vol = train_batch_dir + "/volume-{0}.nii".format(i_vol)
vol_proxy = nib.load(vol)
vol_array = vol_proxy.get_data()

seg = train_batch_dir + "/segmentation-{0}.nii".format(i_vol)
seg_proxy = nib.load(seg)
seg_array = seg_proxy.get_data()

print np.shape(vol_array)
print np.shape(seg_array)

vol_slice = vol_array[:,:,50]
seg_slice = seg_array[:,:,50]

print np.shape(vol_slice)
print np.shape(seg_slice)

X_batch = np.ndarray((1,1,512,512))
Y_batch = np.ndarray((1,1,512,512))

X_batch[0,0,:,:] = vol_slice
Y_batch[0,0,:,:] = seg_slice

img_batch, img_labels = myAugmenter.getAugmentation(X_batch, Y_batch, [0,1,0])
print np.shape(img_batch)
print np.shape(img_labels)
Image.fromarray(img_batch[0,0,:,:]).show()
Image.fromarray(img_labels[0,0,:,:]).show()
img1.show()
'''
'''