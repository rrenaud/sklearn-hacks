# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sklearn.datasets import fetch_20newsgroups
cats = ['alt.atheism', 'sci.space']
train =  newsgroups_train = fetch_20newsgroups(subset='train', categories=cats,
                                               remove=('headers', 'footers', 'quotes'))

# <codecell>

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)

# <codecell>

from sklearn import metrics
newsgroups_test = fetch_20newsgroups(subset='test', categories=cats, 
                                     remove=('headers', 'footers', 'quotes'))
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = LogisticRegression()
clf.fit(vectors, train.target)

pred = clf.predict(vectors_test)
print pred[0:10]
print metrics.f1_score(newsgroups_test.target, pred, average='weighted')

# <codecell>

vectors.shape[1]
import numpy as np
import numpy.random as r
import scipy.sparse as sp
import math
x = sp.csc_matrix(r.uniform(size=(1,2)))
print type(train.target), type(train.target[0]), type(float(train.target[0]))
del i

# <codecell>

class MyLogReg:
    def pred_prob(self, row):
        raw = self.w.multiply(row).sum()
        exp_raw = math.exp(raw)
        return exp_raw / (exp_raw + 1.)
    
    def pred(self, rows):
        return [self.pred_prob(row) > .8 for row in rows]
    
    def fit(self, data, targets):
        self.w = sp.csc_matrix((1,data.shape[1]))
        print self.w.shape, data.shape[0]
        for j in range(1):
            print self.train_epoch(data, targets)
            
    def train_epoch(self, data, targets):
        log_loss = 0.0
        for i in xrange(data.shape[0]):
            cur_row = data[i]
            prediction = self.pred_prob(cur_row)
            #print type(targets[i])
            error = targets[i] - prediction
            log_loss += math.log(prediction if targets[i] else 1 - prediction)
            delta_w = error * cur_row
            self.w = self.w + delta_w
        return log_loss / data.shape[0]      
            
my_log_reg = MyLogReg()
my_log_reg.fit(vectors, train.target)
my_preds = my_log_reg.pred(newsgroups_test.target)
print 'test size is', newsgroups_test.target.shape
print metrics.f1_score(newsgroups_test.target, my_preds, average='weighted')

