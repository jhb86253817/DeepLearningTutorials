"""
This script preprocess the corpus, transform each doc to lsi vector and then store it in json file with ndarray
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
from nltk.corpus import stopwords
import re
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora, models, similarities
import numpy as np
from random import shuffle
import cPickle

def doc2vec(papers):
    #filter non-words
    word_re = r"\w+"
    papers = [[w.lower() for w in nltk.regexp_tokenize(paper,word_re)] for paper in papers]
    #get rid of stopwords
    en_stopwords = set(stopwords.words('english'))
    papers = [[w for w in paper if not w in en_stopwords] for paper in papers]
    #stemming
    #st = LancasterStemmer()
    #papers = [[st.stem(w) for w in paper] for paper in papers]
    #get rid of words that only occur once
    words_all = sum(papers,[])
    rare_words = set([w for w in set(words_all) if words_all.count(w)==1])
    papers = [[w for w in paper if not w in rare_words] for paper in papers]
    #build LSI model for papers
    dictionary = corpora.Dictionary(papers)
    corpus = [dictionary.doc2bow(paper) for paper in papers]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    index = similarities.MatrixSimilarity(lsi[corpus])
    corpus_lsi = [lsi[c] for c in corpus]
    corpus_lsi = [[w[1] for w in c] for c in corpus_lsi]
    return corpus_lsi

def list2array(data):
    data_feature = [f for f,l in data]
    data_labels = [l for f,l in data]
    return (np.array(data_feature), np.array(data_labels))

if __name__ == '__main__':
    with open('arxiv_cs.json', 'rb') as f:
        papers_cs_string = f.read()
    with open('arxiv_math.json', 'rb') as f:
        papers_math_string = f.read()
    papers_cs = json.loads(papers_cs_string)
    papers_math = json.loads(papers_math_string)
    papers_num = 5000
    papers_cs = papers_cs[:papers_num]
    papers_math = papers_math[:papers_num]
    labels = [1] * len(papers_cs) + [0] * len(papers_math)
    corpus_lsi = doc2vec(papers_cs+papers_math)
    data_set = zip(corpus_lsi, labels)
    shuffle(data_set)
    train_set, valid_set, test_set = data_set[:int(papers_num*2*0.7)],\
    data_set[int(papers_num*2*0.7):int(papers_num*2*0.8)],\
    data_set[int(papers_num*2*0.8):] 
    train_set = list2array(train_set)
    valid_set = list2array(valid_set)
    test_set = list2array(test_set)
    with open('arxiv_cs_math_5000_100', 'wb') as f:
        cPickle.dump((train_set,valid_set,test_set), f)


