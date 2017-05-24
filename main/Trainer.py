
import numpy as np


'''
Trains a neural network given the network, a training batch of images and corresponding labels, and a learning rate.
'''

class Trainer:
    def __init__(self, network, network_name, tra_list, val_list, learning_rate=0.01,
                 batch_size=1, patch_size=(220,220)):
        self.network = network
        self.name = network_name
        self.tra_list = tra_list
        self.val_list = val_list
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patch_size = patch_size

    def train_batch(self, X_tra, Y_tra, verbose=False):

        weights_map = np.ndarray(Y_tra.shape)
        weights_map.fill(1)
        if verbose:
            print('training...')
            print 'train X', X_tra.shape, X_tra.dtype, X_tra.min(), X_tra.max(), np.any(np.isnan(X_tra))
            print 'train Y', Y_tra.shape, Y_tra.dtype, Y_tra.min(), Y_tra.max(), np.any(np.isnan(Y_tra))
            print 'train weights_map', weights_map.shape, weights_map.dtype, weights_map.min(), \
                weights_map.max(), np.any(np.isnan(Y_tra))
        loss, l2_loss, accuracy, target_prediction, prediction = \
            self.network.train_fn(X_tra.astype(np.float32), Y_tra.astype(np.int32), weights_map.astype(np.float32))

        return loss, l2_loss, accuracy, target_prediction, prediction

    def validate_batch(self, X_val, Y_val, verbose=False):

        weights_map = np.ndarray(Y_val.shape)
        weights_map.fill(1)
        if verbose:
            print('validation...')
            print 'validation X', X_val.shape, X_val.dtype, X_val.min(), X_val.max(), np.any(np.isnan(X_val))
            print 'validation Y', Y_val.shape, Y_val.dtype, Y_val.min(), Y_val.max(), np.any(np.isnan(Y_val))
            print 'validation weights_map', weights_map.shape, weights_map.dtype, weights_map.min(), \
                weights_map.max(), np.any(np.isnan(Y_val))
        loss, l2_loss, accuracy, target_prediction, prediction = \
            self.network.val_fn(X_val.astype(np.float32), Y_val.astype(np.int32), weights_map.astype(np.float32))

        return loss, l2_loss, accuracy, target_prediction, prediction



