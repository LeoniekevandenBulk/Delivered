
import numpy as np
import nibabel as nib
from tools import *

# Read training data

test_dir = './testSessionLiTS_lesion'
gt_dir = '../../../../LiTS/data/Training_Batch'

# Predicted segmentations
test_pred_segs = sorted(get_file_list(test_dir, 'prediction')[0])
test_pred_segs = [file.replace('\\', '/') for file in test_pred_segs]
# Predicted probabilities
test_pred_probs = [file.replace(test_dir, test_dir+'/ProbMaps') for file in test_pred_segs]
test_pred_probs = [file.replace('Segm', 'ProbMapClass1') for file in test_pred_probs]
# Ground truth segmentations
test_gt_segs = [file.replace(test_dir, gt_dir) for file in test_pred_segs]
test_gt_segs = [file.replace(test_dir, gt_dir) for file in test_pred_segs]
test_gt_segs = [file.replace('prediction', 'segmentation') for file in test_gt_segs]
test_gt_segs = [file.replace('_Segm', '') for file in test_gt_segs]
# Volumes
test_vols = [file.replace('segmentation', 'volume') for file in test_gt_segs]

for i, (vol, seg, pred, prob) in enumerate(zip(test_vols, test_gt_segs, test_pred_segs, test_pred_probs)):
    show_deepMedic_prediction(vol, seg, pred, prob, test_dir + '/show_volumes/')

a