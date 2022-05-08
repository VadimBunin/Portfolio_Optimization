import numpy as np
import pandas as pd
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
print(cov)


w = kit.msr(.02, er, cov)

print(np.round(w, 3))

msr_w = pd.DataFrame(np.round(w, 3), index=er.index, columns=['Weights'])

print(msr_w)
