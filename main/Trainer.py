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

'''
Trainer
'''

class Trainer:

    '''
        Constructor to catch SURFsara-dependent imports
    '''
    def __init__(self, SURFsara, save_network_every_epoch):
        self.SURFsara = SURFsara
        self.save_network_every_epoch = save_network_every_epoch



    '''
        Function to read an existing network from file.
        Input:  - network_name details the name of the file (i.e. network)
        Output: - network that was loaded from memory
    '''
    def readNetwork(self, network_name, patch_size, inputs, targets, weights, depth, branching_factor):
        # Initialize the networks
        unet = UNetClass(inputs, input_size=patch_size,
                             depth=depth,
                             branching_factor=branching_factor,
                             num_input_channels=1,
                             num_classes=2,
                             pad='valid')

        network = unet.define_network()
        
        # Load and set stored parameters from file
        npz = np.load('./' + network_name + '.npz') 
        lasagne.layers.set_all_param_values(network, npz['params'])

        # Create theano functions
        print("Creating Theano training and validation functions...")
        unet.train_fn, unet.val_fn = unet.define_updates(network, inputs, targets, weights)
        unet.predict_fn = unet.define_predict(network, inputs)

        return unet



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
                        patch_size, depth, branching_factor, out_size,
                        img_center, train_batch_dir, inputs, targets,
                        weights, target_class, tra_list, val_list,
                        aug_params, learning_rate, nr_epochs, batch_size,
                        group_percentages, read_slices, slice_files, nr_slices_per_volume,
                        weight_balance, mask_network = None, mask_name = None, mask_threshold = 0):
        
        # Initialize the network        
        unet = UNetClass(inputs, 
					    input_size=patch_size,
					    depth=depth,
					    branching_factor=branching_factor, # 2^6 filters for first level, 2^7 for second, etc.
					    num_input_channels=1,
					    num_classes=2,
					    pad='valid')

        network = unet.define_network()
        
        # Create theano functions
        print("Creating Theano training and validation functions...")
        unet.train_fn, unet.val_fn = unet.define_updates(network, inputs, targets, weights,
                                                                  learning_rate=learning_rate, momentum=0.9, l2_lambda=1e-5)
        unet.predict_fn = unet.define_predict(network, inputs)

        # Define batch generator
        batchGenerator = BatchGenerator(tra_list, val_list, mask_network, mask_name, mask_threshold, train_batch_dir,
                                        target_class, read_slices, slice_files, nr_slices_per_volume, patch_size,
                                        out_size, img_center, group_percentages)

        nr_tra_pixels = np.sum(batchGenerator.seg_tra_slices>=0)
        nr_tra_background = np.sum(batchGenerator.seg_tra_slices == 0)
        nr_tra_labeled = np.sum(batchGenerator.seg_tra_slices == 1)
        fraction_labeled = 1.0 * nr_tra_labeled / nr_tra_pixels
        print("Outof {} training pixels total, {} are background and {} are labeled, which is a fraction of {}").format(
            nr_tra_pixels, nr_tra_background, nr_tra_labeled, fraction_labeled)

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
        test_preprocessing = True
        if test_preprocessing:
            print ('Show preprocessed slices')
            show_preprocessing(batchGenerator, augmenter, aug_params,
                               patch_size, out_size, img_center, target_class)

         # Begin of training
        for epoch in range(nr_epochs):
            print('Epoch {0}/{1}'.format(epoch + 1,nr_epochs))
            
            # Performance per training batch
            tra_loss = []
            tra_dices = []
            tra_ces = []

            # Generate training batch
            for i,batch in enumerate(batchGenerator.get_batch(batch_size, train=True)):
                
                # Assign batc
                X_tra, Y_tra, M_tra = batch

                # Augment data batch
                #X_tra, Y_tra, M_tra = augmenter.getAugmentation(X_tra, Y_tra, M_tra, aug_params)

                # ROI METHOD WERE WE PUT EVERY PIXEL OUTSIDE OF LIVER TO ZERO
                if(target_class == 'lesion'):
                    X_tra[np.where(M_tra == 0)] = np.min(X_tra)
                
                # Pad X and crop Y for UNet, note that array dimensions change here!
                X_tra = batchGenerator.pad(X_tra, patch_size, pad_value=np.min(X_tra))
                Y_tra = batchGenerator.crop(Y_tra, out_size)
                M_tra = batchGenerator.crop(M_tra, out_size)

                #Train and return result for evaluation (reshape to out_size)
                prediction, loss, accuracy = self.train_batch(unet, X_tra, Y_tra, M_tra, weight_balance, target_class)
                prediction = prediction.reshape(batch_size, 1, out_size[0], out_size[1], 2)[:,:,:,:,1]
                    
                # Get Evaluation report
                if i % 100 == 0:
                    print ('batch {}/{}, X min {:.2f} max {:.2f}, Y min {:.2f} max {:.2f}, pred min {:.2f} max {:.2f}'.format(
                        i, (len(tra_list)*nr_slices_per_volume)/batch_size, np.min(X_tra), np.max(X_tra), np.min(Y_tra), np.max(Y_tra),
                        np.min(prediction), np.max(prediction)))
                error_report = evaluator.get_evaluation(Y_tra, prediction)
                dice = error_report[0][1]
                cross_entropy = error_report[1][1]

                # Report and save performance
                if i % 100 == 0:
                    print ('batch {}/{}, {} slc, training loss {:.2f}, dice {:.2f}, cross entropy {:.2f}, cpu {:.2f} min'.format(
                        i, (len(tra_list)*nr_slices_per_volume)/batch_size, batch_size, np.asscalar(loss), dice, cross_entropy, (time.time() - start_time)/60))
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
            for i,batch in enumerate(batchGenerator.get_batch(batch_size, train=False)):
        
                # Assign validation batch
                X_val, Y_val, M_val = batch

                # ROI METHOD WERE WE PUT EVERY PIXEL OUTSIDE OF LIVER TO ZERO
                if(target_class == 'lesion'):
                    X_val[np.where(M_val == 0)] = np.min(X_val)

                # Pad X and crop Y for UNet, note that array dimensions change here!
                X_val = batchGenerator.pad(X_val, patch_size, pad_value=np.min(X_tra))
                Y_val = batchGenerator.crop(Y_val, out_size)
                M_val = batchGenerator.crop(M_val, out_size)
                
                # Get prediction on batch
                prediction, loss, accuracy = self.validate_batch(unet, X_val, Y_val, M_val, weight_balance, target_class)
                prediction = prediction.reshape(batch_size, 1, out_size[0], out_size[1], 2)[:, :, :, :, 1]
                
                # Get evaluation report
                if i % 100 == 0:
                    print ('batch {}/{}, X min {:.2f} max {:.2f}, Y min {:.2f} max {:.2f}, pred min {:.2f} max {:.2f}'.format(
                        i, (len(val_list)*nr_slices_per_volume)/batch_size, np.min(X_tra), np.max(X_tra), np.min(Y_val), np.max(Y_val),
                          np.min(prediction), np.max(prediction)))
                error_report = evaluator.get_evaluation(Y_val, prediction)
                dice = error_report[0][1]
                threshold = error_report[0][2]
                cross_entropy = error_report[1][1]

                # Report and save performance
                if i % 100 == 0:
                    print ('batch {}/{}, {} slc, validation loss {:.2f}, threshold {:.2f}, dice {:.2f}, cross entropy {:.2f}, cpu {:.2f} min'.format(
                        i, (len(val_list)*nr_slices_per_volume)/batch_size, batch_size, np.asscalar(loss), threshold, dice, cross_entropy, (time.time() - start_time)/60))
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
                best_val_dice = np.mean(val_dices)
                # save networks
                params = lasagne.layers.get_all_param_values(network)
                np.savez(os.path.join('./', network_name + '_' + str(best_val_loss) + '_' + str(best_val_dice) + '_' + str(best_val_threshold) + '.npz'), params=params)

            # Save networks every epoch if boolean is on
            if self.save_network_every_epoch:
                params = lasagne.layers.get_all_param_values(network)
                np.savez(
                    os.path.join('./', network_name + '_mini_epoch' + str(epoch) + '_' + str(best_val_threshold) + '.npz'),
                    params=params)

            np.savez('results', tra_loss_lst, tra_dice_lst, tra_ce_lst, val_loss_lst, val_dice_lst, val_ce_lst,
                    best_val_loss, best_val_threshold)

            # Plot result of this epoch
            if epoch == nr_epochs-1:
                self.plotResults(tra_loss_lst, tra_dice_lst, tra_ce_lst, val_loss_lst, val_dice_lst, val_ce_lst,
                    best_val_loss, best_val_threshold)

        return unet, best_val_threshold

    '''
        Function to train network on a batch
        Returns the generated heatmaps to use for evaluation
    '''
    def train_batch(self, unet, X_tra, Y_tra, M_tra, weight_balance, target_class):

        weights_map = np.ndarray(Y_tra.shape)
        weights_map.fill(1)
        
        #if(target_class == 'lesion'):
            # MASK METHOD WERE WE PUT WEIGHTS OUTSIDE OF LIVER TO ZERO
            #weights_map[np.where(M_tra == 0)] = 0

            # ROI METHOD WERE WE PUT EVERY PIXEL OUTSIDE OF LIVER TO ZERO
            #X_tra[np.where(M_tra == 0)] = np.min(X_tra)
            
        weights_map[np.where(Y_tra == 1)] = weight_balance # Labeled pixels can be weigthed more than non-lesion pixels


        loss, l2_loss, accuracy, target_prediction, prediction = \
            unet.train_fn(X_tra.astype(np.float32), Y_tra.astype(np.int32), weights_map.astype(np.float32))

        return prediction, loss, accuracy

    '''
        Function to train network on a batch
        Returns the generated heatmaps to use for evaluation
    '''
    def validate_batch(self, unet, X_val, Y_val, M_val, weight_balance, target_class):

        weights_map = np.ndarray(Y_val.shape)
        weights_map.fill(1)
        
        #if(target_class == 'lesion'):
            # MASK METHOD WERE WE PUT WEIGHTS OUTSIDE OF LIVER TO ZERO
            #weights_map[np.where(M_val == 0)] = 0

            # ROI METHOD WERE WE PUT EVERY PIXEL OUTSIDE OF LIVER TO ZERO
            #X_val[np.where(M_val == 0)] = np.min(X_val)
            
        weights_map[np.where(Y_val == 1)] = weight_balance # Labeled pixels can be weigthed more than non-lesion pixels

        loss, l2_loss, accuracy, target_prediction, prediction = \
            unet.val_fn(X_val.astype(np.float32), Y_val.astype(np.int32), weights_map.astype(np.float32))

        return prediction, loss, accuracy

    '''
        Function to create heatmaps for each image in the batch
    '''
    def predict_batch(self, unet, X_val):

        prediction = unet.predict_fn(X_val.astype(np.float32))

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
        plt.xlabel('mini epoch')
        plt.ylabel('loss')
        plt.ylim([0,0.5])
        plt.legend([tra_loss_plt, val_loss_plt], #, tra_ce_plt, val_ce_plt],
               ['training loss', 'validation loss'], #, 'training cross_entropy', 'validation cross_entropy'],
               loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Best validation loss = {:.2f} with threshold {:.2f}'.format(best_val_loss,
                best_val_threshold), size=40)
        #plt.show(block=False)
        #plt.pause(.5)

        # Save plot        
        plt.savefig('Performance.png')

