import os 
import numpy as np
import pylab as plt
#mnist module 
from mnist import MNIST

#Careful with working directory. Use 'os' to ensure that you are in the correct one.


#Import and load data.

mdata = MNIST('data_mnist')
images_training, labels_training = mdata.load_training()
images_testing, labels_testing = mdata.load_testing()

#Finds the nearest neighbour in the training set for a given image
def nearest_neib_pos(train_set,im_test):
    #returns the rank in training set of the nearest vector of im_test
    norm_list = [np.linalg.norm(np.array(train_set[i])-np.array(im_test)) for i in range(len(train_set))]
    return(np.argmin(np.array(norm_list)))



#Try the model on test set
successes = 0
for idx_t,image_t in enumerate(images_testing[0:100]):
    idx_NN = nearest_neib_pos(images_training,image_t)
    lab_train = labels_training[idx_NN]
    lab_test = labels_testing[idx_t]
    if lab_train == lab_test:
        successes += 1
