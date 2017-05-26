import os
import lasagne
import numpy as np
np.set_printoptions(precision=2, suppress=True)

from UNetClass import UNetClass
from BatchGenerator import BatchGenerator
from BatchAugmenter import BatchAugmenter
from Evaluator import Evaluator


class Trainer:

    '''
        Constructor to catch SURFsara-dependent imports
    '''     
    def __init__(self, SURFsara):
        self.SURFsara = SURFsara        

        import matplotlib
        if SURFsara:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib.rcParams['figure.figsize'] = (40, 24)
        matplotlib.rcParams['xtick.labelsize'] = 30


    '''
        Function to read an existing network from file.
        Input:  - network_name details the name of the file (i.e. network)
        Output: - network that was loaded from memory
    '''
    def readNetwork(self, network_name, patch_size, inputs, targets, weights, depth, branching_factor):
        # Initialize the networks
        network = UNetClass(inputs, input_size=patch_size,
                             depth=depth,
                             branching_factor=branching_fanctor, 
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
    def trainNetwork(self, network_name, 
                        patch_size, depth, branching_factor, out_size, image_center, 
                         train_batch_dir, inputs, targets, weights,
                        target_class, train_data, val_data, 
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
        tra_dice_lst = []
        val_dice_lst = []
        tra_ce_lst = []
        val_ce_lst = []
        best_val_dice = 0
        best_val_threshold = 0

        for epoch in range(nr_epochs):
            print('Epoch {0}/{1}'.format(epoch + 1,nr_epochs))
            
            # Performance per training batch
            tra_dices = []
            tra_ces = []
                
            # Training loop
            for batch in range(nr_train_batches):
                print('Batch {0}/{1}'.format(batch + 1, nr_batches))
                
                # Generate batch
                X_tra, Y_tra = batchGenerator.get_batch(tra_list, train_batch_dir, batch_size,
                                     patch_size, out_size, img_center, target_class=target_class, 
                                     group_percentages=(0.5,0.5))

                # Augment data batch
                X_tra, Y_tra = augmenter.getAugmentation(X_tra, Y_tra, aug_params)

                # Clip, then apply zero mean std 1 normalization
                X = np.clip(X, -200, 300)
                X = (X - X.mean()) / X.std()

                #Train and return result for evaluation
                prediction = self.train_batch(network, X_tra, Y_tra)

                # Get Evaluation report
                error_report = evaluator.get_evaluation(Y_tra, prediction)
                dice = error_report[0][1]
                cross_entropy = error_report[1][1]

                # Report and save performance
                print 'training dice, cross entropy', dice, cross_entropy
                tra_dices.append(dice)
                tra_ces.append(cross_entropy)
            # End training loop
                

            # Performance per validation batch
            val_dices= []
            val_ces= []
            val_thres = []

            # Validation loop
            for batch in range(nr_val_batches):
        
                # Generate batch
                X_val, Y_val = batchGenerator.get_batch(val_list, train_batch_dir, batch_size,
                                     patch_size, out_size, img_center, target_class=target_class, 
                                     group_percentages=(0.5,0.5))

                # Get prediction on batch
                prediction = self.predict_batch(X_val, Y_val)

                # Get evaluation report
                error_report = evaluator.get_evaluation(Y_val, prediction)
                dice = error_report[0][1]
                threshold = error_report[0][2]
                cross_entropy = error_report[1][1]

                # Report and save performance
                print 'validation dice, threshold, cross entropy', dice, threshold, cross_entropy 
                val_dices.append(dice)
                val_ces.append(cross_entropy)
                val_thres.append(threshold)
            # End validation loop


            # Average performance of batches and save
            tra_dice_lst.append(np.mean(tra_dices))
            tra_ce_lst.append(np.mean(tra_ces))
            val_dice_lst.append(np.mean(val_dices))
            val_ce_lst.append(np.mean(val_ces))

            # If performance is best, save network
            if np.mean(val_dices) > best_val_dice:
                best_val_dice = np.mean(val_dices)
                best_val_threshold = np.mean(val_thres)
                # save networks
                params = L.get_all_param_values(liverNetwork.net)
                np.savez(os.path.join('./', network_name + '_' + best_val_dice + '_' + best_val_threshold + '.npz'), params=params)

            # Plot result of this epoch
            self.plotResults(tra_dice_lst, tra_ce_lst, val_dice_lst, val_ce_lst, 
                best_val_dice, best_val_treshold)


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

        return prediction


    '''
        Function to create heatmaps for each image in the batch
    '''
    def predict_batch(self, X_val, Y_val):

        weights_map = np.ndarray(Y_val.shape)
        weights_map.fill(1)

        if self.SURFsara:
            prediction = self.network.predict_fn(X_val.astype(np.float64))
        else:            
            prediction = self.network.predict_fn(X_val.astype(np.float32))

        return prediction


    '''
        Plot the results of the training and validation so far
    '''
    def plotResults(self, tra_dice_lst, tra_ce_lst, val_dice_lst, val_ce_lst, 
                best_val_dice, best_val_treshold):
          
        # plot learning curves
        fig = plt.figure(figsize=(30, 15))
        plt.xlabel('epoch', size=40)
        plt.ylabel('cross_entropy', size=40)
        fig.labelsize=40

        # plot learning curves
        tra_dice_plt, = plt.plot(range(len(tra_dice_lst)), tra_dice_lst, 'b')
        val_dice_plt, = plt.plot(range(len(val_dice_lst)), val_dice_lst, 'g')
        tra_ce_plt, = plt.plot(range(len(tra_ce_lst)), tra_ce_lst, 'm')
        val_ce_plt, = plt.plot(range(len(val_ce_lst)), val_ce_lst, 'r')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend([tra_dice_plt, val_dice_plt, tra_ce_plt, val_ce_plt],
               ['training dice', 'validation dice', 'training cross_entropy', 'validation cross_entropy'],
               loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Best validation dice = {:.2f}% with threshold {:d}'.format(best_val_dice,
                best_val_threshold), size=40)
        plt.show(block=False)
        plt.pause(.5)

        # Save plot        
        plt.savefig('Performance.png')

