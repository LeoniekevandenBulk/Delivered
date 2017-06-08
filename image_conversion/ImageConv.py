'''
Loads nifti files and converts them to jpg files
'''

import numpy as np
from tools import *


'''
Data loading
'''

train_batch_dir='../data/Training_Batch'

vol_batch = sorted(get_file_list(train_batch_dir, 'volume')[0])
seg_batch = sorted(get_file_list(train_batch_dir, 'segmentation')[0])

show_vol = False
if show_vol:
        show_volumes(vol_batch, seg_batch)

for i, (vol, seg) in enumerate(zip(vol_batch, seg_batch)):
        vol_proxy = nib.load(vol)
        vol_array = vol_proxy.get_data()
        print vol_array
        seg_proxy = nib.load(seg)
        seg_array = seg_proxy.get_data()

print vol_batch
