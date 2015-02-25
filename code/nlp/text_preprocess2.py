"""
This script preprocess the corpus, transform each sentence to a list of word
vectors and then store it in json file with ndarray
format.

train_set, valid_set, test_set format: tuple(input, target)
input is an numpy.ndarray of 2 dimensions (a matrix)
witch row's correspond to an example. target is a
numpy.ndarray of 1 dimensions (vector)) that have the same length as
the number of rows in the input. It should give the target
target to the example with the same index in the input.
"""

import json
import nltk
import re
import numpy as np
import cPickle
import theano
import theano.tensor as T

if __name__ == "__main__":
    emb = theano.shared(value=0.2*np.random.uniform(-1.0,1.0,(4,4)).astype(theano.config.floatX))
    idxs = T.ivector()
    x = emb[idxs]
    sample = [0,0,1,2,3]
    f = theano.function(inputs=[idxs], outputs=x)
    print f(sample)

