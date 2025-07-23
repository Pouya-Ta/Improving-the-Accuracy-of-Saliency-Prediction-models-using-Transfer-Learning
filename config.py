import math

#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# batch size
b_s = 10
# number of rows of input images
shape_r = 224
# number of cols of input images
shape_c = 224
# number of rows of predicted maps
shape_r_gt = int(math.ceil(shape_r / 8))
# number of cols of predicted maps
shape_c_gt = int(math.ceil(shape_c / 8))
# number of epochs
nb_epoch = 20

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training images
imgs_train_path = '/kaggle/input/salicon/train_npys/data/processed/images/train/'
# path of training maps
maps_train_path = '/kaggle/input/salicon/train_npys/data/processed/maps/train/'
# number of training images
nb_imgs_train = 700
# path of validation images
imgs_val_path = '/kaggle/input/salicon/val_npys/data/processed/images/val/'
# path of validation maps
maps_val_path = '/kaggle/input/salicon/val_npys/data/processed/maps/val/'
# number of validation images
nb_imgs_val = 150