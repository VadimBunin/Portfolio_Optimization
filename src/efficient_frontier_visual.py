import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kit
from scipy.optimize import minimize

df = pd.read_csv("data/Momentum_Pred.csv", index_col=0)


stocks = pd.DataFrame(df)


er = stocks.T
print(er.shape)


sr = pd.read_csv("data/Stocks_returns.csv")
history_winners = sr[['ASHG.TA', 'BEZQ.TA', 'ENOG.TA',
                      'ESLT.TA', 'ICL.TA', 'LUMI.TA']]

cov = history_winners.cov()
print(cov)

er_v = er.values
print(er_v.shape)
er_w = np.squeeze(er_v)
print(er_w.shape)
kit.plot_ef(20, er_w, cov)
plt.show()
