import kit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import yfinance as yf


# winners = ['ASHG.TA','BEZQ.TA', 'ENOG.TA', 'ESLT.TA',
# 'ICL.TA', 'KEN.TA', 'LUMI.TA', 'TSEM.TA']

start = "2022-03-01"
end = "2022-03-31"
df = yf.download('TA35.TA', start, end)
df.to_csv("data/TA35.TA.csv")
df = pd.read_csv("data/TA35.TA.csv", header=0, index_col=0, parse_dates=[0])
# print(df.head())

close = df.Close.copy()
returns = close.pct_change().copy()
returns.dropna(inplace=True)
vol = np.std(returns)*(np.sqrt(21))
print(vol)
df = returns.resample('M').apply(kit.compound).to_period('M')
print(df)
