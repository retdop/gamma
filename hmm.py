import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

os.chdir('/home/gabriel/fun/trading/gamma')
#os.chdir('/Users/hubert/Documents/D/Bit')


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
overlap = 5

ncomp = 11

Y = []
Ybis = []
i = 0
while i < len(X):
    if i < len(X)-lon:
        res = []
        for j in range(lon):
            res.append(X[i+j])
        
        Ybis.append(res)
        res = normalize2(res)
        Y.append(res)
        i = i+overlap
    else:
        i += 1

Y = np.array(Y)
Ybis = np.array(Ybis)


df = pd.DataFrame(Y)
m = np.zeros(len(df))
v = np.zeros(len(df))
for i in range(len(m)):
    m[i] = np.mean(Y[i])
    v[i] = np.var(Y[i])
    #vol[i] = sqrt(np.var())
    
df['mean'] = m
df['var'] = v
lbd = np.int(2)
model = GaussianHMM(n_components = ncomp).fit(np.concatenate((Y[0:lbd*1000], Y[(lbd + 1)*1000:])),
                    [np.int(lbd*1000), np.int(len(Y) - (lbd+1)*1000)])


hidden_states = model.predict(Y)
hs = []
for i in range(len(hidden_states)):
    for j in range(ncomp):
        hs.append(hidden_states[i])

W = model.transmat_


#%%

states = pd.DataFrame(np.zeros((ncomp, 5)), columns=['count', 'avg_variation', 'avg_mean', 'avg_variance', 'avg_vol'])

for i in range(ncomp):
    ind = np.where(hidden_states == i)[0]
    print('State', i)
    states.iloc[i]['count'] = len(ind)/len(Y)*100
    print('Count',len(ind), '{:0.2f} %'.format(states.ix[i,'count']))
    states.iloc[i]['avg_variation'] = (np.mean([Y.T[-1][j]/Y.T[-6][j] for j in np.where(hidden_states == i)[0]]) - 1) * 100
    print('Average variation {:0.2f} %'.format(states.ix[i, 'avg_variation']))
    states.iloc[i]['avg_mean'] = np.mean(df['mean'].ix[np.where(hidden_states == i)])
    print('Average mean {:0.5f}'.format(states.ix[i,'avg_mean']))
    states.iloc[i]['avg_variance'] = np.mean(df['var'].ix[np.where(hidden_states == i)])
    print('Average variance {:0.7f}'.format(states.ix[i, 'avg_variance']))
    states.iloc[i]['avg_vol'] = np.mean(df['var'].ix[np.where(hidden_states == i)])
    print('Average volatility {:0.7f}'.format(states.ix[i, 'avg_vol']))
    for i in ind:
        plt.plot(Y[i])
    plt.show()
#%% Strategy test
    
Ytest = Y[9000:]
Ybistest = Ybis[9000:]

Spred = model.predict(Ytest)

bets = np.zeros(len(Ytest))

for i in range(len(bets)):
    u = Spred[i]
    bets[i] = sum([W[u][j] * 1 * (states.loc[j, 'avg_variation']) for j in range(ncomp)])
    
earnings = [
        1 * (Ybistest[i+1][14] / Ybistest[i+1][9] - 1)
        for i in range(len(bets) - 1) if bets[i] > 0.03]
print(sum(earnings))
plt.plot(earnings)
