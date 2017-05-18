'''
Trains a neural network given the network, a training batch of images and corresponding labels, and a learning rate.
'''

class Trainer:
    def __init__(self, network, network_name, tra_batch, tra_labels, learning_rate):
        self.network = network
        self.name = network_name
        self.tra_batch = tra_batch
        self.tra_labels = tra_labels
        self.learning_rate = learning_rate




