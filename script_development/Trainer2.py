class Trainer:


    '''
        Function to read an existing network from file.
        Input:  - network_name details the name of the file (i.e. network)
        Output: - network that was loaded from memory
    '''
    def readNetwork(self, network_name):
        # Initialize the networks
        network = UNetClass(inputs, input_size=patch_size,
                             depth=5,
                             branching_factor=3,  # 2^6 filters for first level, 2^7 for second, etc.
                             num_input_channels=1,
                             num_classes=2,
                             pad='valid')

        # Load and set stored parameters from file
        npz = np.load('./' + network_name + '.npz') 
        lasagne.layers.set_all_param_values(network.net, npz['params'])

        # Create theano functions
        print("Creating Theano training and validation functions...")
        network.train_fn, network.val_fn = network.define_updates(inputs, targets, weights)

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
    def trainNetwork(self, network_name, target_class, train_data, val_data):
        
        # Initialisze the network        
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

        # Define batch generator
        batchGenerator = BatchGenerator()

        # Define data augmenter
        augmenter = BatchAugmenter(X_tra, Y_tra, [[0.1,0.8,0.9],[0.1,0.8,0.6]])
        
        # Define variables to keep score of performance
        tra_cross_entropy = []
        tra_dice = []
        val_cross_entropy = []
        val_dice = []
        best_val_acc = 0

        for epoch in range(nr_epochs):
            print('Epoch {0}/{1}'.format(epoch + 1,nr_epochs))
            
            # Training loop
            for batch in range(nr_train_batches):
                print('Batch {0}/{1}'.format(batch + 1, nr_batches))
                
                # Generate batch
                X_tra, Y_tra = batchGenerator.get_batch(tra_list, train_batch_dir, batch_size,
                                     patch_size, out_size, img_center, target_class=target_class, 
                                     group_percentages=(0.5,0.5))

                # Augment data batch
                #X_tra, Y_tra = augmenter.getAugmentation()

                # Clip, then apply zero mean std 1 normalization
                X = np.clip(X, -200, 300)
                X = (X - X.mean()) / X.std()

                #Train and return result for evaluation
                prediction = self.train_batch(network, X_tra, Y_tra)

                



            # Validation loop
            for batch in range(nr_val_batches):
        
                # Generate batch
                X_val, Y_val = batchGenerator.get_batch(val_list, train_batch_dir, batch_size,
                                     patch_size, out_size, img_center, target_class=target_class, 
                                     group_percentages=(0.5,0.5))

                # Get prediction on batch
                prediction = self.validate_batch(X_val, Y_val)


            # Plot result of this epoch
            self.plotResults(tra_cross_entropy, tra_dice,
                            val_cross_entropy, val_dice)


    '''
        Function to train network on a batch
        Returns the generated heatmaps to use for evaluation
    '''
    def train_batch(self, network, X_tra, Y_tra):

        weights_map = np.ndarray(Y_tra.shape)
        weights_map.fill(1)
        loss, l2_loss, accuracy, target_prediction, prediction = \
        network.train_fn(X_tra.astype(np.float32), Y_tra.astype(np.int32), weights_map.astype(np.float32))

        return prediction


    '''
        Plot the results of the training and validation so far
    '''
    def plotResults(self, tra_cross_entropy, tra_dice, val_cross_entropy, val_dice):
          
        # plot learning curves
        fig = plt.figure(figsize=(30, 15))
        plt.xlabel('epoch', size=40)
        plt.ylabel('cross_entropy', size=40)
        fig.labelsize=40

        # plot learning curves
            tra_loss_plt, = plt.plot(range(len(tra_loss_lst)), tra_loss_lst, 'b')
            val_loss_plt, = plt.plot(range(len(val_loss_lst)), val_loss_lst, 'g')
            tra_acc_plt, = plt.plot(range(len(tra_acc_lst)), tra_acc_lst, 'm')
            val_acc_plt, = plt.plot(range(len(val_acc_lst)), val_acc_lst, 'r')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend([tra_loss_plt, val_loss_plt, tra_acc_plt, val_acc_plt],
                       ['training loss', 'validation loss', 'training accuracy', 'validation accuracy'],
                       loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('Best validation accuracy = {:.2f}%'.format(100. * best_val_acc), size=40)
            plt.show(block=False)
            plt.pause(.5)

        # Save plot        
        plt.savefig('Performance.png')



    if np.mean(val_accs) > best_val_acc:
        best_val_acc = np.mean(val_accs)
        # save networks
        params = L.get_all_param_values(liverNetwork.net)
        np.savez(os.path.join('./', liver_network_name + '.npz'), params=params)
        params = L.get_all_param_values(lesionNetwork.net)
        np.savez(os.path.join('./', lesion_network_name + '.npz'), params=params)

