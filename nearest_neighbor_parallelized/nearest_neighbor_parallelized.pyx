# --- Adapted from Stanford course CS231N ----
#  http://cs231n.github.io/

import numpy as np
from utils import load_CIFAR10 

cimport numpy as np
cimport cython
from libc.math cimport log2 # FIXME: can be unavailable in Window
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref, preincrement as inc
from cython.parallel import prange


cdef class NearestNeighbor:
    
    cdef double[:,:] Xtr
    cdef long[:] ytr

    def __init__(self):
        pass

    def train(self, np.ndarray[double, ndim=2] X, np.ndarray[long, ndim=1] y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    @cython.boundscheck(False) # turn of bounds-checking for entire function
    @cython.cdivision(True)
    cpdef np.ndarray[long, ndim=1] predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        cdef int num_test = X.shape[0]

        # lets make sure that the output type matches the input type
        cdef np.ndarray[long,ndim=1] Ypred = np.empty(num_test, dtype = np.int)
        cdef int i
        # loop over all test rows

        for i in prange(num_test, nogil=True, schedule='dynamic'):
            with gil:
                if i%1000 == 0:
                    print i, num_test
                # find the nearest training image to the i'th test image
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i]), axis = 1)
                min_index = np.argmin(distances) # get the index with smallest distance
                Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

        return Ypred


def classify():
    cdef long[:] Yte_predict 

    Xtr, Ytr, Xte, Yte = load_CIFAR10('/home/phamorim/Downloads/cifar-10-batches-py/')
    # flatten out all images to be one-dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
    nn = NearestNeighbor() # create a Nearest Neighbor classifier class
    nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
    
    Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
