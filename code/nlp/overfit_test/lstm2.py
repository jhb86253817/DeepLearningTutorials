# LSTM without peepholes
# Adagrad
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
        self.wxg = theano.shared(name='wxg',
                                value=0.2 * numpy.random.uniform(1.0, -1.0, (nw, nh))
                                .astype(theano.config.floatX))
        self.whg = theano.shared(name='whg',
                                value=0.2 * numpy.random.uniform(1.0, -1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.wxi = theano.shared(name='wxi',
                                value=0.2 * numpy.random.uniform(1.0, -1.0, (nw, nh))
                                .astype(theano.config.floatX))
        self.whi = theano.shared(name='whi',
                                value=0.2 * numpy.random.uniform(1.0, -1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.wxf = theano.shared(name='wxf',
                                value=0.2 * numpy.random.uniform(1.0, -1.0, (nw, nh))
                                .astype(theano.config.floatX))
        self.whf = theano.shared(name='whf',
                                value=0.2 * numpy.random.uniform(1.0, -1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.wxo = theano.shared(name='wxo',
                                value=0.2 * numpy.random.uniform(1.0, -1.0, (nw, nh))
                                .astype(theano.config.floatX))
        self.who = theano.shared(name='who',
                                value=0.2 * numpy.random.uniform(1.0, -1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                                value=0.2 * numpy.random.uniform(1.0, -1.0, (nh, nw))
                               .astype(theano.config.floatX))
        self.bg = theano.shared(name='bg',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bi = theano.shared(name='bi',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bf = theano.shared(name='bf',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo = theano.shared(name='bo',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nw,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.c0 = theano.shared(name='c0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        self.wxg_acc = theano.shared(name='wxg_acc',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.whg_acc = theano.shared(name='whg_acc',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wxi_acc = theano.shared(name='wxi_acc',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.whi_acc = theano.shared(name='whi_acc',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wxf_acc = theano.shared(name='wxf_acc',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.whf_acc = theano.shared(name='whf_acc',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.wxo_acc = theano.shared(name='wxo_acc',
                                value=numpy.zeros((nw, nh),
                                dtype=theano.config.floatX))
        self.who_acc = theano.shared(name='who_acc',
                                value=numpy.zeros((nh, nh),
                                dtype=theano.config.floatX))
        self.w_acc = theano.shared(name='w_acc',
                                value=numpy.zeros((nh, nw),
                                dtype=theano.config.floatX))
        self.bg_acc = theano.shared(name='bg_acc',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bi_acc = theano.shared(name='bi_acc',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bf_acc = theano.shared(name='bf_acc',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo_acc = theano.shared(name='bo_acc',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b_acc = theano.shared(name='b_acc',
                               value=numpy.zeros(nw,
                               dtype=theano.config.floatX))
        self.h0_acc = theano.shared(name='h0_acc',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.c0_acc = theano.shared(name='c0_acc',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))


        #bundle
        self.params = [self.wxg, self.whg, self.wxi, self.whi, self.wxf, self.whf, self.wxo, self.who, self.w, self.bg, self.bi, self.bf, self.bo, self.b, self.h0, self.c0]
        self.params_acc = [self.wxg_acc, self.whg_acc, self.wxi_acc, self.whi_acc, self.wxf_acc, self.whf_acc, self.wxo_acc, self.who_acc, self.w_acc, self.bg_acc, self.bi_acc, self.bf_acc, self.bo_acc, self.b_acc, self.h0_acc, self.c0_acc]

        idxs = T.ivector()
        x = self.index[idxs]
        y_sentence = T.ivector('y_sentence') # labels

        def recurrence(x_t, c_tm1, h_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wxi) + T.dot(h_tm1, self.whi) + self.bi)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wxf) + T.dot(h_tm1, self.whf) + self.bf)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wxo) + T.dot(h_tm1, self.who) + self.bo)
            g_t = T.tanh(T.dot(x_t, self.wxg) + T.dot(h_tm1, self.whg) + self.bg)
            c_t = f_t * c_tm1 + i_t * g_t
            h_t = o_t * T.tanh(c_t)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [c_t, h_t, s_t]

        [c, h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.c0, self.h0, None],
                                n_steps=x.shape[0],
                                truncate_gradient=-1)

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log2(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])

        sentence_gradients = [T.grad(sentence_nll, param) for param in self.params]
        #Adagrad
        sentence_updates = []
        for param_i, grad_i, acc_i in zip(self.params, sentence_gradients, self.params_acc):
            acc = acc_i + T.sqr(grad_i)
            sentence_updates.append((param_i, param_i - lr*grad_i/(T.sqrt(acc)+1e-5)))
            sentence_updates.append((acc_i, acc))

        # SGD
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
        for param in self.params+self.params_acc:
            numpy.save(os.path.join(folder, param.name+'.npy'),
                    param.get_value())

    def load(self, folder):
        for param in self.params+self.params_acc:
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
    ppls = [rnn.ppl(x,y) for (x,y) in zip(data[0][:100], data[1][:100])]
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
                'lr': 3,
            'nhidden': 50,
            # number of hidden units
            'seed': 345,
            'nepochs': 60,
            # 60 is recommended
            'savemodel': True,
            'loadmodel': False,
            'folder':'lstm2_40_10_0.1',
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
    #random.seed(param['seed'])

    rnn = RNNLM(nh=param['nhidden'],
                nw=len(train_dict))

    if param['word2vec'] == True:
        rnn.load_word2vec()

    # load parameters
    if param['loadmodel'] == True:
        print "loading parameters\n"
        rnn.load(param['folder'])

    if param['train'] == True:

        round_num =  40
        train_lines = 10
        #adapt learning rate
        #lrs = [param['lr'] * (1 - iter_num/(round_num*train_lines)) for iter_num in xrange(1,round_num*train_lines+1)]

        train_data_labels = zip(train_data[0], train_data[1])
        print "Training..."
        start = time.time()

        i = 1
        for j in xrange(round_num):
            random.shuffle(train_data_labels)
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