import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

os.chdir('/home/gabriel/fun/trading/gamma')
data = pd.read_csv('data/data_all.txt', delimiter = ' ')

X = data['Weighted_Price'].as_matrix()[:-1]

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
