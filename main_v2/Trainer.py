import os
import lasagne
import numpy as np
np.set_printoptions(precision=2, suppress=True)

from UNetClass import UNetClass
from BatchGenerator import BatchGenerator
from BatchAugmenter import BatchAugmenter
from Evaluator import Evaluator

from tools import *

import time

SURFsara = False

import matplotlib
if SURFsara:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (40, 24)
matplotlib.rcParams['xtick.labelsize'] = 30

class Trainer:

    '''
        Constructor to catch SURFsara-dependent imports
    '''
    def __init__(self, SURFsara):
        self.SURFsara = SURFsara

    '''
        Function to read an existing network from file.
        Input:  - network_name details the name of the file (i.e. network)
        Output: - network that was loaded from memory
    '''
    def readNetwork(self, network_name, patch_size, inputs, targets, weights, depth, branching_factor):
        # Initialize the networks
        network = UNetClass(inputs, input_size=patch_size,
                             depth=depth,
                             branching_factor=branching_factor,
                             num_input_channels=1,
                             num_classes=2,
                             pad='valid')

        # Load and set stored parameters from file
        npz = np.load('./' + network_name + '.npz') 
        lasagne.layers.set_all_param_values(network.net, npz['params'])

        # Create theano functions
        print("Creating Theano training and validation functions...")
        network.train_fn, network.val_fn = network.define_updates(inputs, targets, weights)
        network.predict_fn = network.define_predict(inputs)

        return network



    '''
        Function that trains a network on training data, and report on progress during 
        validation on validation data

        Input:  - network_name details the name of the file (i.e. network)
                - train_data contains the volumes to train on
                - val_data contains the volumes to validate on
        Output: - network is a trained network
                - threshold is the best threshold for the heatmap of the network
    '''
    def trainNetwork(self, start_time, network_name, mask,
                        patch_size, depth, branching_factor, out_size, img_center,
                        train_batch_dir, inputs, targets, weights,
                        target_class, tra_list, val_list,
                        aug_params, learning_rate,
                        nr_epochs, nr_train_batches, nr_val_batches, batch_size
,                       mask_network = None, threshold = 0):
        
        # Initialize the network        
        network = UNetClass(inputs, 
					    input_size=patch_size,
					    depth=depth,
					    branching_factor=branching_factor, # 2^6 filters for first level, 2^7 for second, etc.
					    num_input_channels=1,
					    num_classes=2,
					    pad='valid')

        # Create theano functions
        print("Creating Theano training and validation functions...")
        network.train_fn, network.val_fn = network.define_updates(inputs, targets, weights)
        network.predict_fn = network.define_predict(inputs)

        # Define batch generator
        batchGenerator = BatchGenerator(mask_network, threshold)

        # Define data augmenter
        augmenter = BatchAugmenter()

        # Define evaluator
        evaluator = Evaluator()
        
        # Define variables to keep score of performance
        tra_loss_lst = []
        val_loss_lst = []
        tra_dice_lst = []
        val_dice_lst = []
        tra_ce_lst = []
        val_ce_lst = []
        best_val_loss = 10000
        best_val_threshold = 0

        # Test de BatchGenerator en pre-processing steps
        test_preprocessing = False
        if test_preprocessing:
            show_preprocessing(tra_list, train_batch_dir, batchGenerator, patch_size, out_size, img_center)

        for epoch in range(nr_epochs):
            print('Epoch {0}/{1}'.format(epoch + 1,nr_epochs))
            
            # Performance per training batch
            tra_loss = []
            tra_dices = []
            tra_ces = []
                
            # Training loop
            for batch in range(nr_train_batches):
                print('Batch {0}/{1}'.format(batch + 1, nr_train_batches))
                
                # Generate batch
                X_tra, Y_tra, M_tra = batchGenerator.get_batch(tra_list, train_batch_dir, batch_size,
                                     patch_size, out_size, img_center, target_class=target_class, 
                                     group_percentages=(0.5,0.5))

                # Augment data batch
                X_tra, Y_tra = augmenter.getAugmentation(X_tra, Y_tra, aug_params)

                # Clip, then apply zero mean std 1 normalization
                X_tra = np.clip(X_tra, -200, 300)
                #X_tra = (X_tra - X_tra.mean()) / X_tra.std()
                X_tra = (X_tra + 200) / 500

                # If lesion netwerk, then apply mask
                if target_class == 'lesion':
                    if mask == 'liver':
                        X_mask = mask_network.predict(X_tra)
                    elif mask == 'ground truth':
                        X_mask = M_tra # temporarily use ground truth mask for lesion network
                        X_tra = (X_mask > 0).astype(np.int32) * X_tra

                # Pad X and crop Y for UNet, note that array dimensions change here!
                X_tra, Y_tra = batchGenerator.pad_and_crop(X_tra, Y_tra, patch_size, out_size, img_center)

                #Train and return result for evaluation (reshape to out_size)
                prediction, loss, accuracy = self.train_batch(network, X_tra, Y_tra)
                prediction = prediction.reshape(batch_size, 1, out_size[0], out_size[1], 2)[:,:,:,:,1]

                # Get Evaluation report
                error_report = evaluator.get_evaluation(Y_tra, prediction)
                dice = error_report[0][1]
                cross_entropy = error_report[1][1]

                # Report and save performance
                print ('training loss {:.2f}, dice {:.2f}, cross entropy {:.2f}, cpu {:.2f} min'.format( \
                    np.asscalar(loss), dice, cross_entropy, (time.time() - start_time)/60))
                tra_loss.append(loss)
                tra_dices.append(dice)
                tra_ces.append(cross_entropy)
            # End training loop
                

            # Performance per validation batch
            val_loss= []
            val_dices= []
            val_ces= []
            val_thres = []

            # Validation loop
            for batch in range(nr_val_batches):
        
                # Generate batch
                X_val, Y_val, M_val = batchGenerator.get_batch(val_list, train_batch_dir, batch_size,
                                     patch_size, out_size, img_center, target_class=target_class, 
                                     group_percentages=(0.5,0.5))

                # Clip, then apply zero mean std 1 normalization
                X_val = np.clip(X_val, -200, 300)
                #X_val = (X_val - X_val.mean()) / X_val.std()
                X_val = (X_val + 200) / 500

                # If lesion netwerk, then apply mask
                if target_class == 'lesion':
                    if mask == 'liver':
                        X_mask = mask_network.predict(X_val)
                    elif mask == 'ground truth':
                        X_mask = M_val # temporarily use ground truth mask for lesion network
                        X_val = (X_mask > 0).astype(np.int32) * X_val

                # Pad X and crop Y for UNet, note that array dimensions change here!
                X_val, Y_val = batchGenerator.pad_and_crop(X_val, Y_val, patch_size, out_size, img_center)

                # Get prediction on batch
                prediction, loss, accuracy = self.validate_batch(network, X_val, Y_val)
                prediction = prediction.reshape(batch_size, 1, out_size[0], out_size[1], 2)[:, :, :, :, 1]


                # Get evaluation report
                error_report = evaluator.get_evaluation(Y_val, prediction)
                dice = error_report[0][1]
                threshold = error_report[0][2]
                cross_entropy = error_report[1][1]

                # Report and save performance
                print ('validation loss {:.2f}, threshold {:.2f}, dice {:.2f}, cross entropy {:.2f}, cpu {:.2f} min'.format( \
                    np.asscalar(loss), threshold, dice, cross_entropy, (time.time() - start_time)/60))
                val_loss.append(loss)
                val_dices.append(dice)
                val_ces.append(cross_entropy)
                val_thres.append(threshold)
            # End validation loop


            # Average performance of batches and save
            tra_loss_lst.append(np.mean(tra_loss))
            tra_dice_lst.append(np.mean(tra_dices))
            tra_ce_lst.append(np.mean(tra_ces))
            val_loss_lst.append(np.mean(val_loss))
            val_dice_lst.append(np.mean(val_dices))
            val_ce_lst.append(np.mean(val_ces))

            # If performance is best, save network
            if np.mean(val_loss) < best_val_loss:
                best_val_loss = np.mean(val_loss)
                best_val_threshold = np.mean(val_thres)
                # save networks
                params = lasagne.layers.get_all_param_values(network.net)
                np.savez(os.path.join('../../Networks/test/', network_name + '_' + str(best_val_loss) + '_' + str(best_val_threshold) + '.npz'), params=params)

            # Plot result of this epoch
            if not self.SURFsara and epoch == nr_epochs-1:
                self.plotResults(tra_loss_lst, tra_dice_lst, tra_ce_lst, val_loss_lst, val_dice_lst, val_ce_lst,
                    best_val_loss, best_val_threshold)

        return network, best_val_threshold

    '''
        Function to train network on a batch
        Returns the generated heatmaps to use for evaluation
    '''
    def train_batch(self, network, X_tra, Y_tra):

        weights_map = np.ndarray(Y_tra.shape)
        weights_map.fill(1)

        if self.SURFsara:
            loss, l2_loss, accuracy, target_prediction, prediction = \
            network.train_fn(X_tra.astype(np.float64), Y_tra.astype(np.int32), weights_map.astype(np.float64))
        else:
            loss, l2_loss, accuracy, target_prediction, prediction = \
            network.train_fn(X_tra.astype(np.float32), Y_tra.astype(np.int32), weights_map.astype(np.float32))

        return prediction, loss, accuracy

    '''
        Function to train network on a batch
        Returns the generated heatmaps to use for evaluation
    '''
    def validate_batch(self, network, X_val, Y_val):

        weights_map = np.ndarray(Y_val.shape)
        weights_map.fill(1)

        if self.SURFsara:
            loss, l2_loss, accuracy, target_prediction, prediction = \
            network.val_fn(X_val.astype(np.float64), Y_val.astype(np.int32), weights_map.astype(np.float64))
        else:
            loss, l2_loss, accuracy, target_prediction, prediction = \
            network.val_fn(X_val.astype(np.float32), Y_val.astype(np.int32), weights_map.astype(np.float32))

        return prediction, loss, accuracy

    '''
        Function to create heatmaps for each image in the batch
    '''
    def predict_batch(self, network, X_val, Y_val):

        weights_map = np.ndarray(Y_val.shape)
        weights_map.fill(1)

        if self.SURFsara:
            prediction = network.predict_fn(X_val.astype(np.float64))
        else:            
            prediction = network.predict_fn(X_val.astype(np.float32))

        return prediction[0]


    '''
        Plot the results of the training and validation so far
    '''
    def plotResults(self, tra_loss_lst, tra_dice_lst, tra_ce_lst, val_loss_lst, val_dice_lst, val_ce_lst,
                best_val_loss, best_val_threshold):

        # plot learning curves
        fig = plt.figure(figsize=(30, 15))
        plt.xlabel('epoch', size=40)
        plt.ylabel('cross_entropy', size=40)
        fig.labelsize=40

        # plot learning curves
        tra_loss_plt, = plt.plot(range(len(tra_loss_lst)), tra_loss_lst, 'b')
        val_loss_plt, = plt.plot(range(len(val_loss_lst)), val_loss_lst, 'g')
        #tra_dice_plt, = plt.plot(range(len(tra_dice_lst)), tra_dice_lst, 'b')
        #val_dice_plt, = plt.plot(range(len(val_dice_lst)), val_dice_lst, 'g')
        #tra_ce_plt, = plt.plot(range(len(tra_ce_lst)), tra_ce_lst, 'm')
        #val_ce_plt, = plt.plot(range(len(val_ce_lst)), val_ce_lst, 'r')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend([tra_loss_plt, val_loss_plt], #, tra_ce_plt, val_ce_plt],
               ['training loss', 'validation loss'], #, 'training cross_entropy', 'validation cross_entropy'],
               loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Best validation loss = {:.2f}% with threshold {:.2f}'.format(best_val_loss,
                best_val_threshold), size=40)
        plt.show(block=False)
        plt.pause(.5)

        # Save plot        
        plt.savefig('Performance.png')

