from calendar import month
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import yfinance as yf
import kit
print("hello world")


returns = pd.read_csv('data/Stocks_returns.csv',
                      index_col=0, parse_dates=True)

#m_return = kit.compound(returns)

df = returns.resample('M').apply(kit.compound).to_period('M')
print(df.head())
