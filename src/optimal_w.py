import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kit


# winners =

df = pd.read_csv("data/Momentum.csv", index_col=0)
# print(df)

stocks = pd.DataFrame(df)

# print(stocks.T)

er = stocks.T


sr = pd.read_csv("data/Stocks_returns.csv")
history_winners = sr[['ASHG.TA', 'BEZQ.TA', 'ENOG.TA',
                      'ESLT.TA', 'ICL.TA', 'KEN.TA', 'LUMI.TA', 'TSEM.TA']]

cov = history_winners.cov()
# print(cov)

er_v = er.values
er_w = np.squeeze(er_v)
kit.plot_ef(20, er_w, cov)
plt.show()
