import scipy.io
import numpy as np

img_width, img_height = 224, 224
num_channels = 3
batch_size = 32

cars_meta = scipy.io.loadmat('./datasets/devkit/cars_meta')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)
num_classes = len(class_names)

char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'