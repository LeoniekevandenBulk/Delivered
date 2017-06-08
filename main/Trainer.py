import os
import lasagne
import numpy as np
np.set_printoptions(precision=2, suppress=True)

SURFsara = True

import matplotlib
if SURFsara:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from UNetClass import UNetClass
from BatchGenerator import BatchGenerator
from BatchAugmenter import BatchAugmenter
from Evaluator import Evaluator

from tools import *

import time



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
                        nr_epochs, nr_train_batches, nr_val_batches, batch_size,
                        read_slices, slice_files,
                        mask_network = None, threshold = 0):
        
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
        network.train_fn, network.val_fn = network.define_updates(inputs, targets, weights,
                                                                  learning_rate=learning_rate, momentum=0.9, l2_lambda=1e-5)
        network.predict_fn = network.define_predict(inputs)

        # Define batch generator
        batchGenerator = BatchGenerator(mask_network, threshold, tra_list, val_list, train_batch_dir, target_class,
                                        read_slices, slice_files,
                                        group_percentages=(0.5, 0.5), nr_slices_per_volume=100)

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
        best_val_loss = 1
        best_val_threshold = 0


        # Test de BatchGenerator en pre-processing steps
        test_preprocessing = True
        if test_preprocessing:
            print ('Show preprocessed slices')
            show_preprocessing(batchGenerator, augmenter, aug_params, \
                               patch_size, out_size, img_center, target_class)

        for epoch in range(nr_epochs):
            print('Epoch {0}/{1}'.format(epoch + 1,nr_epochs))
            
            # Performance per training batch
            tra_loss = []
            tra_dices = []
            tra_ces = []

            # Training loop
            for batch in range(nr_train_batches):
                #print('Batch {0}/{1}'.format(batch + 1, nr_train_batches))
                
                # Generate batch
                X_tra, Y_tra = batchGenerator.get_train_batch(batch_size)
                #print ('get batch X min {:.2f} max {:.2f}, Y min {:.2f} max {:.2f}'.format(
                #      np.min(X_tra), np.max(X_tra), np.min(Y_tra), np.max(Y_tra)))
                # Augment data batch
                #X_tra, Y_tra = augmenter.getAugmentation(X_tra, Y_tra, aug_params)

                # Pad X and crop Y for UNet, note that array dimensions change here!
                X_tra, Y_tra = batchGenerator.pad_and_crop(X_tra, Y_tra, patch_size, out_size, img_center)
                #print ('pad & crop X min {:.2f} max {:.2f}, Y min {:.2f} max {:.2f}'.format(
                #      np.min(X_tra), np.max(X_tra), np.min(Y_tra), np.max(Y_tra)))

                #Train and return result for evaluation (reshape to out_size)
                prediction, loss, accuracy = self.train_batch(network, X_tra, Y_tra)
                prediction = prediction.reshape(batch_size, 1, out_size[0], out_size[1], 2)[:,:,:,:,1]

                # Get Evaluation report
                if batch%100 == 0:
                    print ('batch {}/{}, X min {:.2f} max {:.2f}, Y min {:.2f} max {:.2f}, pred min {:.2f} max {:.2f}'.format(
                        batch, nr_train_batches, np.min(X_tra), np.max(X_tra), np.min(Y_tra), np.max(Y_tra),
                        np.min(prediction), np.max(prediction)))
                error_report = evaluator.get_evaluation(Y_tra, prediction)
                dice = error_report[0][1]
                cross_entropy = error_report[1][1]

                # Report and save performance
                if batch%100 == 0:
                    print ('batch {}/{}, {} slc, training loss {:.2f}, dice {:.2f}, cross entropy {:.2f}, cpu {:.2f} min'.format( \
                        batch, nr_train_batches, batch_size, np.asscalar(loss), dice, cross_entropy, (time.time() - start_time)/60))
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
                X_val, Y_val = batchGenerator.get_val_batch(batch_size)

                # Pad X and crop Y for UNet, note that array dimensions change here!
                X_val, Y_val = batchGenerator.pad_and_crop(X_val, Y_val, patch_size, out_size, img_center)

                # Get prediction on batch
                prediction, loss, accuracy = self.validate_batch(network, X_val, Y_val)
                prediction = prediction.reshape(batch_size, 1, out_size[0], out_size[1], 2)[:, :, :, :, 1]


                # Get evaluation report
                if batch % 100 == 0:
                    print ('batch {}/{}, X min {:.2f} max {:.2f}, Y min {:.2f} max {:.2f}, pred min {:.2f} max {:.2f}'.format(
                        batch, nr_val_batches, np.min(X_tra), np.max(X_tra), np.min(Y_val), np.max(Y_val),
                          np.min(prediction), np.max(prediction)))
                error_report = evaluator.get_evaluation(Y_val, prediction)
                dice = error_report[0][1]
                threshold = error_report[0][2]
                cross_entropy = error_report[1][1]

                # Report and save performance
                if batch % 100 == 0:
                    print ('batch {}/{}, {} slc, validation loss {:.2f}, threshold {:.2f}, dice {:.2f}, cross entropy {:.2f}, cpu {:.2f} min'.format( \
                        batch, nr_val_batches, batch_size, np.asscalar(loss), threshold, dice, cross_entropy, (time.time() - start_time)/60))
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
                np.savez(os.path.join('./', network_name + '_' + str(best_val_loss) + '_' + str(best_val_threshold) + '.npz'), params=params)

            np.savez('results', tra_loss_lst, tra_dice_lst, tra_ce_lst, val_loss_lst, val_dice_lst, val_ce_lst,
                    best_val_loss, best_val_threshold)

            # Plot result of this epoch
            if epoch == nr_epochs-1:
                self.plotResults(tra_loss_lst, tra_dice_lst, tra_ce_lst, val_loss_lst, val_dice_lst, val_ce_lst,
                    best_val_loss, best_val_threshold)

        return network, best_val_threshold

    '''
        Function to train network on a batch
        Returns the generated heatmaps to use for evaluation
    '''
    def train_batch(self, network, X_tra, Y_tra):

        weights_map = Y_tra + 0.1 # Lesion pixels are weigthed 100 times more than non-lesion pixels
        #weights_map = np.ndarray(Y_tra.shape)
        #weights_map.fill(1)

        loss, l2_loss, accuracy, target_prediction, prediction = \
            network.train_fn(X_tra.astype(np.float32), Y_tra.astype(np.int32), weights_map.astype(np.float32))

        return prediction, loss, accuracy

    '''
        Function to train network on a batch
        Returns the generated heatmaps to use for evaluation
    '''
    def validate_batch(self, network, X_val, Y_val):

        weights_map = Y_val + 0.1 # Lesion pixels are weigthed 100 times more than non-lesion pixels
        #weights_map = np.ndarray(Y_tra.shape)
        #weights_map.fill(1)

        loss, l2_loss, accuracy, target_prediction, prediction = \
            network.val_fn(X_val.astype(np.float32), Y_val.astype(np.int32), weights_map.astype(np.float32))

        return prediction, loss, accuracy

    '''
        Function to create heatmaps for each image in the batch
    '''
    def predict_batch(self, network, X_val, Y_val):

        weights_map = np.ndarray(Y_val.shape)
        weights_map.fill(1)

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
        plt.ylim([0,0.25])
        plt.legend([tra_loss_plt, val_loss_plt], #, tra_ce_plt, val_ce_plt],
               ['training loss', 'validation loss'], #, 'training cross_entropy', 'validation cross_entropy'],
               loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Best validation loss = {:.2f} with threshold {:.2f}'.format(best_val_loss,
                best_val_threshold), size=40)
        #plt.show(block=False)
        #plt.pause(.5)

        # Save plot        
        plt.savefig('Performance.png')

