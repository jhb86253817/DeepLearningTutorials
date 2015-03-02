from gensim.models import Word2Vec
import numpy
import theano
import json


if __name__ == '__main__':
    with open('../ptb.train.txt', 'rb') as f:
        sentences = [line.strip() for line in f]
    sentences = [['<bos>']+line.split() for line in sentences]
    model = Word2Vec(sentences, size=50, min_count=0)
    with open('../index2word.json', 'rb') as ff:
        index2word = json.loads(ff.read())
    vocab = [index2word[str(index)] for index in xrange(10000)]
    wx_vec = [model[w] for w in vocab]
    wx_vec = numpy.array(wx_vec)
    wx = theano.shared(name='wx', value=wx_vec.astype(theano.config.floatX))
    numpy.save('wx.npy', wx.get_value())


    
