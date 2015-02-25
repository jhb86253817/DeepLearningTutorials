#from collections import defaultdict
#
#wids = defaultdict(lambda: len(wids))
#s = 'i am implementing model recurrent neural language model'
#s = s.split()
#for w in s:
#    print wids[w]
#assert len(wids) == 7 

import theano
import theano.tensor as T
import numpy as np

# defining the tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym),
        sequences=X)
compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=[results])

# test values
x = np.eye(8, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2

#print compute_elementwise(x, w, b)[0]

# comparison with numpy
#print np.tanh(x.dot(w) + b)

u = np.zeros((1,8), dtype=theano.config.floatX)
x = np.concatenate((u,x),axis=0)
print x
