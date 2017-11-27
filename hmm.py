# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:21:08 2017

@author: hubert
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

#os.chdir('/home/gabriel/fun/trading/gamma')
os.chdir('/Users/hubert/Documents/D/Bit')


data = pd.read_csv('data/data_all.txt', delimiter = ' ')

for col in ['Open', 'High', 'Low', 'Close', 'Volume_BTC', 'Volume_Currency', 'Weighted_Price']:
    print(col)
    data[col] = data[col].astype(float)

X = data['Weighted_Price'].as_matrix()[:-1]

    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def normalize2(v):
    return v / v[0]

#%%
lon = 15
overlap = 1

ncomp = 5

Y = []
i = 0
while i < len(X):
    if i < len(X)-lon:
        res = []
        for j in range(lon):
            res.append(X[i+j])
        
        res = normalize2(res)
        Y.append(res)
        i = i+overlap
    else:
        i += 1

Y = np.array(Y)

df = pd.DataFrame(Y)
m = np.zeros(len(df))
v = np.zeros(len(df))
for i in range(len(m)):
    m[i] = np.mean(Y[i])
    v[i] = np.var(Y[i])
df['mean'] = m
df['var'] = v

model = GaussianHMM(n_components = ncomp).fit(Y)


hidden_states = model.predict(Y)
hs = []
for i in range(len(hidden_states)):
    for j in range(ncomp):
        hs.append(hidden_states[i])

W = model.transmat_


#%%

state = 3

cunt = 0
for i in np.where(hidden_states == state)[0]:
    plt.plot(Y[i])
    cunt += 1
print(cunt)
#%%

print(np.mean(Y.T[-1][np.where(hidden_states == 0)]))
print(np.mean(Y.T[-1][np.where(hidden_states == 1)]))
print(np.mean(Y.T[-1][np.where(hidden_states == 2)]))
