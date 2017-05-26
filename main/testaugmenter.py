from BatchAugmenter import *
import numpy as np
from PIL import Image
from tools import get_file_list

train_batch_dir='../data/Training_Batch'

vol_batch = sorted(get_file_list(train_batch_dir, 'volume')[0])
seg_batch = sorted(get_file_list(train_batch_dir, 'segmentation')[0])

print(np.asarray(vol_batch).shape, np.asarray(seg_batch).shape)

'''
img1 = Image.open("liver.jpg")
img2 = Image.open("liver.jpg")
img_batch = [np.array(img1)[:,:,1], np.array(img2)[:,:,1]]
img_labels = [np.array(img1)[:,:,1], np.array(img2)[:,:,1]]
myAugmenter = BatchAugmenter(img_batch, img_labels, [[0,1,0],[0,1,0]])
img_batch, img_labels = myAugmenter.getAugmentation()
print np.shape(img_batch)
print np.shape(img_labels)
Image.fromarray(img_batch[:,:,0]).show()
Image.fromarray(img_labels[:,:,0]).show()
img1.show()
'''
