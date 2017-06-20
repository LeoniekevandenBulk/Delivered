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
    def trainNetwork(self, start_time, network_name,
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
                               patch_size, out_size, img_center, target_class, mask_name)

        # Default threshold
        threshold = 0.5

        # Start with validaiton for graph!
        val_loss= []
        val_dices= []
        val_ces= []
        val_thres = []

        # Tracking parameters for dice
        tp = 0.0
        fp = 0.0
        fn = 0.0
        
        print('Validate before training!')
        # Validation loop
        for i,batch in enumerate(batchGenerator.get_batch(batch_size, train=False)):
        
            # Assign validation batch
            X_val, Y_val, M_val = batch

            # ROI METHOD WERE WE PUT EVERY PIXEL OUTSIDE OF LIVER TO ZERO
            if(target_class == 'lesion' and not(mask_name == None)):
                X_val[np.where(M_val == 0)] = np.min(X_val)

            # Pad X and crop Y for UNet, note that array dimensions change here!
            X_val = batchGenerator.pad(X_val, patch_size, pad_value=np.min(X_val))
            Y_val = batchGenerator.crop(Y_val, out_size)
            M_val = batchGenerator.crop(M_val, out_size)
                
            # Get prediction on batch
            prediction, loss, accuracy = self.validate_batch(unet, X_val, Y_val, M_val, weight_balance, target_class)
            prediction = prediction.reshape(batch_size, 1, out_size[0], out_size[1], 2)[:, :, :, :, 1]

            # Determine misclassifications for dice score
            thresh_pred = (prediction >= threshold).astype(int)
                
            tp += np.sum( (thresh_pred == 1) & (Y_val == 1) )
            fp += np.sum( (thresh_pred == 1) & (Y_val == 0) )
            fn += np.sum( (thresh_pred == 0) & (Y_val == 1) )
            
            # Get evaluation report
            if i % 100 == 0:
                print ('batch {}/{}, X min {:.2f} max {:.2f}, Y min {:.2f} max {:.2f}, pred min {:.2f} max {:.2f}'.format(
                        i, (len(val_list)*nr_slices_per_volume)/batch_size, np.min(X_val), np.max(X_val), np.min(Y_val), np.max(Y_val),
                        np.min(prediction), np.max(prediction)))
            error_report = evaluator.get_evaluation(Y_val, prediction, threshold)
            dice = error_report[0][1]
            cross_entropy = error_report[1][1]

            # Report and save performance
            if i % 100 == 0:
               print ('batch {}/{}, {} slc, validation loss {:.2f}, threshold {:.2f}, dice {:.2f}, cross entropy {:.2f}, cpu {:.2f} min'.format(
                   i, (len(val_list)*nr_slices_per_volume)/batch_size, batch_size, np.asscalar(loss), threshold, dice, cross_entropy, (time.time() - start_time)/60))
            val_loss.append(loss)
            val_ces.append(cross_entropy)
            if(not(dice==-1)):
                val_dices.append(dice)
                val_thres.append(threshold)

        # Determine dice score over entire vaidation set
        full_val_dice = (2*tp)/(2*tp + fp + fn)
        
        # Average performance of batches and save
        val_loss_lst.append(np.mean(val_loss))
        val_dice_lst.append(full_val_dice)
        val_ce_lst.append(np.mean(val_ces))
        tra_loss_lst.append(np.mean(val_loss))
        tra_dice_lst.append(full_val_dice)
        tra_ce_lst.append(np.mean(val_ces))

        # Begin of training
        for epoch in range(nr_epochs):
            print('Epoch {0}/{1}'.format(epoch + 1,nr_epochs))
            
            # Performance per training batch
            tra_loss = []
            tra_dices = []
            tra_ces = []

            # Tracking parameters for dice
            tp = 0.0
            fp = 0.0
            fn = 0.0

            # Generate training batch
            for i,batch in enumerate(batchGenerator.get_batch(batch_size, train=True)):
                
                # Assign batc
                X_tra, Y_tra, M_tra = batch

                # Augment data batch
                X_tra, Y_tra, M_tra = augmenter.getAugmentation(X_tra, Y_tra, M_tra, aug_params, gauss_percent=0.05)

                # ROI METHOD WERE WE PUT EVERY PIXEL OUTSIDE OF LIVER TO ZERO
                if(target_class == 'lesion' and not(mask_name == None)):
                    X_tra[np.where(M_tra == 0)] = np.min(X_tra)
                
                # Pad X and crop Y for UNet, note that array dimensions change here!
                X_tra = batchGenerator.pad(X_tra, patch_size, pad_value=np.min(X_tra))
                Y_tra = batchGenerator.crop(Y_tra, out_size)
                M_tra = batchGenerator.crop(M_tra, out_size)

                #Train and return result for evaluation (reshape to out_size)
                prediction, loss, accuracy = self.train_batch(unet, X_tra, Y_tra, M_tra, weight_balance, target_class)
                prediction = prediction.reshape(batch_size, 1, out_size[0], out_size[1], 2)[:,:,:,:,1]

                # Determine misclassifications for dice score
                thresh_pred = (prediction >= threshold).astype(int)
                
                tp += np.sum( (thresh_pred == 1) & (Y_tra == 1) )
                fp += np.sum( (thresh_pred == 1) & (Y_ta == 0) )
                fn += np.sum( (thresh_pred == 0) & (Y_tra == 1) )
                    
                # Get Evaluation report
                if i % 100 == 0:
                    print ('batch {}/{}, X min {:.2f} max {:.2f}, Y min {:.2f} max {:.2f}, pred min {:.2f} max {:.2f}'.format(
                        i, (len(tra_list)*nr_slices_per_volume)/batch_size, np.min(X_tra), np.max(X_tra), np.min(Y_tra), np.max(Y_tra),
                        np.min(prediction), np.max(prediction)))
                error_report = evaluator.get_evaluation(Y_tra, prediction, threshold)
                dice = error_report[0][1]
                cross_entropy = error_report[1][1]

                # Report and save performance
                if i % 100 == 0:
                    print ('batch {}/{}, {} slc, training loss {:.2f}, dice {:.2f}, cross entropy {:.2f}, cpu {:.2f} min'.format(
                        i, (len(tra_list)*nr_slices_per_volume)/batch_size, batch_size, np.asscalar(loss), dice, cross_entropy, (time.time() - start_time)/60))
                tra_loss.append(loss)
                tra_ces.append(cross_entropy)
                if(not(dice==-1)):
                    tra_dices.append(dice)
            # End training loop

            # Determine dice score over entire training set
            full_tra_dice = (2*tp)/(2*tp + fp + fn)  

            print('Training had a mean dice score of {0} with threshold {1}'.format(full_tra_dice, threshold))


            ###############################################
            ###############################################


                
            # Performance per validation batch
            val_loss= []
            val_dices= []
            val_ces= []
            val_thres = []

            # Histogram lists
            zero_hist = [0 for _ in range(100)]
            ones_hist = [0 for _ in range(100)]

            # Tracking parameters for dice
            tp = 0.0
            fp = 0.0
            fn = 0.0
            
            # Validation loop
            for i,batch in enumerate(batchGenerator.get_batch(batch_size, train=False)):
        
                # Assign validation batch
                X_val, Y_val, M_val = batch

                # ROI METHOD WERE WE PUT EVERY PIXEL OUTSIDE OF LIVER TO ZERO
                if(target_class == 'lesion' and not(mask_name == None)):
                    X_val[np.where(M_val == 0)] = np.min(X_val)

                # Pad X and crop Y for UNet, note that array dimensions change here!
                X_val = batchGenerator.pad(X_val, patch_size, pad_value=np.min(X_val))
                Y_val = batchGenerator.crop(Y_val, out_size)
                M_val = batchGenerator.crop(M_val, out_size)
                
                # Get prediction on batch
                prediction, loss, accuracy = self.validate_batch(unet, X_val, Y_val, M_val, weight_balance, target_class)
                prediction = prediction.reshape(batch_size, 1, out_size[0], out_size[1], 2)[:, :, :, :, 1]

                # Determine misclassifications for dice score
                thresh_pred = (prediction >= threshold).astype(int)
                
                tp += np.sum( (thresh_pred == 1) & (Y_val == 1) )
                fp += np.sum( (thresh_pred == 1) & (Y_val == 0) )
                fn += np.sum( (thresh_pred == 0) & (Y_val == 1) )
                
                # Get evaluation report
                if i % 100 == 0:
                    print ('batch {}/{}, X min {:.2f} max {:.2f}, Y min {:.2f} max {:.2f}, pred min {:.2f} max {:.2f}'.format(
                        i, (len(val_list)*nr_slices_per_volume)/batch_size, np.min(X_tra), np.max(X_tra), np.min(Y_val), np.max(Y_val),
                          np.min(prediction), np.max(prediction)))
                error_report = evaluator.get_evaluation(Y_val, prediction, threshold)
                dice = error_report[0][1]
                cross_entropy = error_report[1][1]

                # Report and save performance
                if i % 100 == 0:
                    print ('batch {}/{}, {} slc, validation loss {:.2f}, threshold {:.2f}, dice {:.2f}, cross entropy {:.2f}, cpu {:.2f} min'.format(
                        i, (len(val_list)*nr_slices_per_volume)/batch_size, batch_size, np.asscalar(loss), threshold, dice, cross_entropy, (time.time() - start_time)/60))
                val_loss.append(loss)
                val_ces.append(cross_entropy)
                if(not(dice==-1)):
                    val_dices.append(dice)
                    val_thres.append(threshold)

                # Save info for threshold histogram
                zero_hist = addHistogram(prediction, Y_val, 0, zero_hist)
                ones_hist = addHistogram(prediction, Y_val, 1, ones_hist)
        
            # End validation loop

            # Turn histograms into probabilities and determine threshold
            sum_zero_hist = (sum(zero_hist)*1.0)/100
            zero_hist = [(val*1.0)/sum_zero_hist for val in zero_hist]

            sum_ones_hist = (sum(ones_hist)*1.0)/100
            ones_hist = [(val*1.0)/sum_ones_hist for val in ones_hist]

            threshold = findBestThreshold(zero_hist, ones_hist, 1.0, 1.0)

            # Determine dice score over entire vaidation set
            full_val_dice = (2*tp)/(2*tp + fp + fn)
                  
            print('Validation had a mean dice score of {0} with threshold {1}'.format(full_val_dice, threshold))

            ############################################    


            # Average performance of batches and save
            tra_loss_lst.append(np.mean(tra_loss))
            tra_dice_lst.append(full_tra_dice)
            tra_ce_lst.append(np.mean(tra_ces))
            val_loss_lst.append(np.mean(val_loss))
            val_dice_lst.append(full_val_dice)
            val_ce_lst.append(np.mean(val_ces))

            # If performance is best, save network
            if np.mean(val_loss) < best_val_loss:
                best_val_loss = np.mean(val_loss)
                best_val_threshold = threshold
                best_val_dice = full_val_dice
                
                # Save distribution of classification with threshold
                show_threshold_split(zero_hist, ones_hist, threshold)
                
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
        tra_dice_plt, = plt.plot(range(len(tra_dice_lst)), tra_dice_lst, 'r')
        val_dice_plt, = plt.plot(range(len(val_dice_lst)), val_dice_lst, 'y')
        #tra_ce_plt, = plt.plot(range(len(tra_ce_lst)), tra_ce_lst, 'm')
        #val_ce_plt, = plt.plot(range(len(val_ce_lst)), val_ce_lst, 'r')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0,0.5])
        plt.legend([tra_loss_plt, val_loss_plt, tra_dice_plt, val_dice_plt],
               ['training loss', 'validation loss', 'training dice', 'validation dice'],
               loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Best validation loss = {:.2f} with threshold {:.2f}'.format(best_val_loss,
                best_val_threshold), size=40)
        #plt.show(block=False)
        #plt.pause(.5)

        # Save plot        
        plt.savefig('Performance.png')

