# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import numpy.random as r
import scipy.sparse as sp
import math

cats = ['alt.atheism', 'sci.space']
train =  newsgroups_train = fetch_20newsgroups(
    subset='train', categories=cats,
    remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
newsgroups_test = fetch_20newsgroups(subset='test', categories=cats, 
                                     remove=('headers', 'footers', 'quotes'))
vectors_test = vectorizer.transform(newsgroups_test.data)

clf = LogisticRegression()
clf.fit(vectors, train.target)

pred = clf.predict(vectors_test)
f1 = metrics.f1_score(newsgroups_test.target, pred, average='weighted')
print 'sklearn log reg f1', f1


class MyLogReg:
    def pred_prob(self, row):
        # there must be a better way to get the dot prod?
        raw = self.w.multiply(row).sum()
        exp_raw = math.exp(raw)
        return exp_raw / (exp_raw + 1.)
    
    def pred(self, rows):
        return [self.pred_prob(row) > .5 for row in rows]
    
    def fit(self, data, targets):
        self.w = sp.csc_matrix((1,data.shape[1]))
        for j in range(2):
            print 'epoch', j, 'log loss', self.train_epoch(data, targets)
            
    def train_epoch(self, data, targets):
        log_loss = 0.0
        for i in xrange(data.shape[0]):
            cur_row = data[i]
            prediction = self.pred_prob(cur_row)
            error = targets[i] - prediction
            log_loss += -math.log(prediction if targets[i] else 1 - prediction)
            delta_w = error * cur_row
            self.w = self.w + delta_w
        return log_loss / data.shape[0]      
            
my_log_reg = MyLogReg()
my_log_reg.fit(vectors, train.target)
my_preds = my_log_reg.pred(newsgroups_test.target)
my_f1 = metrics.f1_score(newsgroups_test.target, my_preds, average='weighted')
print 'my log reg f1', my_f1

