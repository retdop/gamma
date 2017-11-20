# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:12:03 2017

@author: hubert
"""


import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hmmlearn.hmm import GaussianHMM


os.chdir('/Users/hubert/Documents/D/Bit')
data = pd.read_csv('krakenBTCCharts.csv', delimiter = ';')

X = data['Weighted Price'].as_matrix()[:-1]

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


#%%
lon = 20
ncomp = 3


Y = []
i = 0
while i < len(X):
    if i < len(X)-lon:
        res = []
        for j in range(lon):
            res.append(X[i+j])
            
       # res = normalize(res)
        Y.append(res)
        i = i+lon
    else:
        i += 1

Y = np.array(Y)
model = GaussianHMM(n_components = ncomp).fit(Y)

hidden_states = model.predict(Y)
hs = []
for i in range(len(hidden_states)):
    for j in range(ncomp):
        hs.append(hidden_states[i])

W = model.transmat_


#%%
state = 0
for i in range(len(Y)):
    if hidden_states[i] == state :
        plt.plot(Y[i])
        


#%%


from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn import linear_model


pls = 15
y =[]
x = []
for i in range(len(X)-pls-1):
    x.append(X[i:i+pls])
    if X[i+pls-1] < X[i+pls]:
        y.append(1)
    else :
        y.append(-1)

x = np.array(x)
y = np.array(y)
    


X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.1, random_state=3)

clf = GradientBoostingClassifier()
#clf = svm.SVC()
#clf = linear_model.LogisticRegression()


clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print("CLassifiation report :\n%s\n" % (
    metrics.classification_report(
        Y_test,
        Y_pred)))



