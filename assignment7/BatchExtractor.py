
import numpy as np
import random

'''
Implement a batch extractor

We now implement a batch extractor. This class is based on the sample extractor, but it just makes mini-batches of
random samples. The mini-batches will be used to train your network, and as we have seen, there will be random data
augmentation there.
'''

class BatchExtractor:
    def __init__(self, sample_extractor, batch_size, class_balancing=False):
        self.batch_size = batch_size
        self.class_balancing = class_balancing
        self.sample_extractor = sample_extractor

    def convert_to_onehot(self, Y,
                          num_classes):  # convert a [b,1,h,w] label tensor to a [b,number_of_categories,h,w] tensor in one_hot form
        one_hot = np.zeros((Y.shape[0], num_classes, Y.shape[2], Y.shape[3]))
        for i in range(num_classes):
            one_hot[:, i:i + 1, :, :] = Y.astype(np.int) == i
        one_hot = one_hot.astype(np.float32)
        return one_hot

    def get_random_batch_balanced(self):
        se = self.sample_extractor

        ps_y, ps_x = se.patch_size

        # Write a function to extract a batch with balanced classes using the previously created patch extractor
        # Use the 'get_random_patch_from_class' function you implemented


        X_batch = np.ndarray((self.batch_size, 1, ps_y, ps_x))
        Y_batch = np.ndarray((self.batch_size, 1, 1, 1))
        for i in range(self.batch_size):
            if random.random() > 0.5:
                label = 0
            else:
                label = 1
            X_sample, Y_sample = se.get_random_sample_from_class(label)
            X_batch[i, :, :, :] = X_sample
            Y_batch[i, 0, 0, 0] = Y_sample

        # X_batch should be a 4D array indexed as [batchsize, channels, y, x]
        # Y_batch should be a 4D adday indexed as [batchsize, 1, 1, 1] and will be [batchsize, 2, 1, 1] after onehot conversion
        # the batch should contains 50% positive patches and 50% negative patches

        X_batch = X_batch.astype(np.float32)
        return X_batch, self.convert_to_onehot(Y_batch, 2)