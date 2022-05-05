import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import yfinance as yf
print("hello world")


returns = pd.read_csv('data/Stocks_returns.csv', index_col=0, parse_dates=True)
