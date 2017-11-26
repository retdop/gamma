import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hmmlearn.hmm import GaussianHMM


os.chdir('/home/gabriel/fun/trading/gamma')
data = pd.read_csv('data/data_all.txt', delimiter = ' ')
for col in ['Open', 'High', 'Low', 'Close', 'Volume_BTC', 'Volume_Currency', 'Weighted_Price']:
    data[col] = data[col].astype(float)

X = data['Weighted_Price'].as_matrix()[:-1]

    

from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn import linear_model


pls = 50
print(pls)
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
