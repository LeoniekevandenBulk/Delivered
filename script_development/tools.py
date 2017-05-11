
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# function to get a list of file of a given extension, both the absolute path and the filename

def get_file_list(path,ext='',queue=''):
    if ext != '':
        return [os.path.join(path,f) for f in os.listdir(path) if f.startswith(ext)],  \
               [f for f in os.listdir(path) if f.startswith(ext)]
    else:
        return [os.path.join(path,f) for f in os.listdir(path)]

# Define a function to visualize (1) the fundus image, (2) the binary mask, (3) the manual annotation of a case with a given index
def show_image(idx, imgs, msks, lbls):
    img = np.asarray(Image.open(imgs[idx]))
    print
    img.shape
    msk = np.asarray(Image.open(msks[idx]))
    lbl = np.asarray(Image.open(lbls[idx]))
    plt.subplot(1, 3, 1)
    plt.imshow(img);
    plt.title('RGB image {}'.format(idx + 1))
    plt.subplot(1, 3, 2)
    plt.imshow(msk, cmap='gray');
    plt.title('Mask {}'.format(idx + 1))
    plt.subplot(1, 3, 3)
    plt.imshow(lbl, cmap='gray');
    plt.title('Manual annotation {}'.format(idx + 1))
    plt.show()