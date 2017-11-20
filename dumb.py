import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir('/home/gabriel/fun/trading/gamma')

raw = pd.read_csv('data/krakenBTCCharts.csv', delimiter=';', parse_dates=[0])

print(raw)
raw.describe()

raw['diff'] = raw['Close'] - raw['Open']
raw['diff'].plot()
plt.show()

raw['mean_diff'] = 0.1*(sum([raw['diff'].shift(i) for i in range(0,9)]))

raw['pred'] = 0
mask_up = raw['diff'].shift(1) > 0.5
mask_down = raw['diff'].shift(1) < -0.5
raw['pred'][mask_up] = 1
raw['pred'][mask_down] = -1

def good_prediction(raw):
    print(raw[(raw['pred'] * raw['diff'] > 0)]['diff'].count()/raw['diff'].count()*100)
    print(raw[(raw['pred'] * raw['diff'] > 0)]['diff'].count()/raw['diff'].count()*100)

good_prediction(raw)
