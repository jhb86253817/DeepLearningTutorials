# Remove network bias
from __future__ import division
import os
import time
import json
from collections import defaultdict
from collections import OrderedDict
from scipy import stats
import random
import numpy
import theano
from theano import tensor as T

class RNNLM(object):
    """recurrent neural network language model"""
    def __init__(self, nh, nw):
        """
        nh :: dimension of the hidden layer
        nw :: vocabulary size
        """
        # parameters of the model
        self.index = theano.shared(name='index',
                                value=numpy.eye(nw,
                                dtype=theano.config.floatX))
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0, (nw, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nw))
                               .astype(theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        #bundle
        self.params = [self.wx, self.wh, self.w, self.h0]

        idxs = T.ivector()
        x = self.index[idxs]
        y_sentence = T.ivector('y_sentence') # labels

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx)
                                 + T.dot(h_tm1, self.wh))
            s_t = T.nnet.softmax(T.dot(h_t, self.w))
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0],
                                truncate_gradient=-1)

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log2(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])

        sentence_gradients = [T.grad(sentence_nll, param) for param in self.params]
        sentence_updates = [(param, param - lr*g) for param,g in zip(self.params, sentence_gradients)]

        # perplexity of a sentence
        sentence_ppl = T.pow(2, sentence_nll)

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred, allow_input_downcast=True)
        self.prob_dist = theano.function(inputs=[idxs], outputs=p_y_given_x_sentence, allow_input_downcast=True)
        self.ppl = theano.function(inputs=[idxs, y_sentence], outputs=sentence_ppl, allow_input_downcast=True)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates,
                                              allow_input_downcast=True)
    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder, param.name+'.npy'),
                    param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))

    def load_word2vec(self):
        self.wx.set_value(numpy.load('./word2vec/wx.npy'))


def load_data():
    train_file = open('ptb.train.txt', 'r')
    # training set, a list of sentences
    train_set = [l.strip() for l in train_file]
    train_file.close()
    # a list of lists of tokens
    train_set = [l.split() for l in train_set]
    train_dict = defaultdict(lambda: len(train_dict))
    # an extra symbol for the end of a sentence
    train_dict['<bos>'] = 0
    train_labels = [[train_dict[w] for w in l] for l in train_set]
    train_idxs = [[0]+l[:-1] for l in train_labels]
    # transform data and label list to numpy array
    train_idxs = [numpy.array(l) for l in train_idxs]
    train_labels = [numpy.array(l) for l in train_labels]

    valid_file = open('ptb.valid.txt', 'r')
    # validation set, a list of sentences
    valid_set = [l.strip() for l in valid_file]
    valid_file.close()
    # a list of lists of tokens
    valid_set = [l.split() for l in valid_set]
    valid_labels = [[train_dict[w] for w in l] for l in valid_set]
    valid_idxs = [[0]+l[:-1] for l in valid_labels]
    # transform data and label list to numpy array
    valid_idxs = [numpy.array(l) for l in valid_idxs]
    valid_labels = [numpy.array(l) for l in valid_labels]

    test_file = open('ptb.test.txt', 'r')
    # test set, a list of sentences
    test_set = [l.strip() for l in test_file]
    test_file.close()
    # a list of lists of tokens
    test_set = [l.split() for l in test_set]
    test_labels = [[train_dict[w] for w in l] for l in test_set]
    test_idxs = [[0]+l[:-1] for l in test_labels]
    # transform data and label list to numpy array
    test_idxs = [numpy.array(l) for l in test_idxs]
    test_labels = [numpy.array(l) for l in test_labels]

    train_data = (train_idxs, train_labels)
    valid_data = (valid_idxs, valid_labels)
    test_data = (test_idxs, test_labels)

    return train_data, valid_data, test_data, train_dict

def ppl(data, rnn):
    ppls = [rnn.ppl(x,y) for (x,y) in zip(data[0][:10], data[1][:10])]
    mean_ppl = numpy.mean(list(ppls))

    return mean_ppl

def random_generator(probs):
    xk = xrange(10000)
    custm = stats.rv_discrete(name='custm', values=(xk,probs))
    return custm.rvs(size=1)

def next_word(text, train_dict, index2word, rnn, length):
    words = text.split()
    for j in xrange(20):
        idxs = [train_dict[w] for w in words]
        for i in xrange(length):
            prob_dist = rnn.prob_dist(numpy.asarray(idxs).astype('int32'))
            next_index = random_generator(prob_dist[-1,:])
            idxs.append(next_index[0])
        print [index2word[index] for index in idxs]

def main(param=None):
    if not param:
        param = {
            #'lr': 0.0970806646812754,
            #'lr': 3.6970806646812754,
            'lr': 0.1,
            'nhidden': 50,
            # number of hidden units
            'seed': 345,
            'nepochs': 60,
            # 60 is recommended
            'savemodel': True,
            'loadmodel': False,
            'folder':'rnnlm2_40_10_0.1',
            'train': True,
            'test': False,
            'word2vec': False}
    print param

    # load data and dictionary
    train_data, valid_data, test_data, train_dict = load_data()

    #for toy test 
    toy_data = (test_data[0][:5], test_data[1][:5])
    
    #index2word
    index2word = dict([(v,k) for k,v in train_dict.iteritems()])

    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    rnn = RNNLM(nh=param['nhidden'],
                nw=len(train_dict))

    if param['word2vec'] == True:
        rnn.load_word2vec()

    # load parameters
    if param['loadmodel'] == True:
        print "loading parameters\n"
        rnn.load(param['folder'])

    if param['train'] == True:

        round_num = 40 
        train_lines = 10
        #adapt learning rate
        #lrs = [param['lr'] * (1 - iter_num/(round_num*train_lines)) for iter_num in xrange(1,round_num*train_lines+1)]

        train_data_labels = zip(train_data[0], train_data[1])
        print "Training..."
        start = time.time()

        i = 1
        for j in xrange(round_num):
            #random.shuffle(train_data_labels)
            for (x,y) in train_data_labels[:train_lines]:
                rnn.sentence_train(x, y, param['lr'])
                if i%10 == 0:
                    print "%d of %d" % (i, round_num*train_lines)
                    test_ppl = ppl(train_data, rnn)
                    print "Test perplexity of toy data: %f \n" % test_ppl
                i += 1

        #test_ppl = ppl(test_data, rnn)
        #print "Test perplexity of test data: %f \n" % test_ppl

        end = time.time()
        print "%f seconds in total\n" % (end-start)

        # save parameters
        if param['savemodel'] == True:
            print "saving parameters\n"
            rnn.save(param['folder'])

    if param['test'] == True:
        text = "<bos> japan is"
        next_word(text, train_dict, index2word, rnn, 10)

if __name__ == '__main__':
    main()
