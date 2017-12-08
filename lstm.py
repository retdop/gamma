"""Lstm strategy"""

#import numpy as np
import os
import datetime
import pandas as pd
#import matplotlib.pyplot as plt
os.chdir('/home/gabriel/fun/trading/gamma')
#os.chdir('/Users/hubert/Documents/D/Bit')

btc = pd.read_csv('data/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv', delimiter=',')
btc['Date'] = btc['Timestamp'].apply(lambda x: str(datetime.datetime.fromtimestamp(x)))
btc['Volatility'] = btc['Open'].rolling(window=60, center=False).std()
btc['Volatility_averaged'] = btc['Volatility'].rolling(window=300, center=False).mean()

#%%

btc['Open'].plot()
btc['Volatility'].plot()
btc['Volatility_averaged'].plot()
