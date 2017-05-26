from Trainer2 import *

liver_segmentation_name = 'liver_network_LiTS'
load_liver_segmentation = False

lesion_detection_name = 'lesion_network_LiTS'
load_lesion_detection = False


# Create class to train (or load) networks
trainer = Trainer()


# Load or train liver segmentation network
print "Liver Network..."
if (load_liver_segmentation):
    liver_network = trainer.readNetwork(liver_segmentation_name)
else:
    liver_network = trainer.trainNetwork(liver_segmentation_name)


# Load or train lesion detection network
print "Lesion Network..."
if (load_lesion_detection):
    lesion_network = trainer.loadNetwork(lesion_detection_name)
else:
    lesion_network = trainer.trainNetwork(lesion_detection_name)
