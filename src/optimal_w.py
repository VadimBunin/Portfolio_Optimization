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
                      'ESLT.TA', 'ICL.TA', 'LUMI.TA']]

cov = history_winners.cov()
print(cov)

er_v = er.values
er_w = np.squeeze(er_v)
kit.plot_ef(20, er_w, cov)
plt.show()


weights = kit.optimal_weights(20, er_w, cov)
# print(weights)

vs = pd.DataFrame(np.round(weights, 3), columns=['ASHG.TA', 'BEZQ.TA', 'ENOG.TA',
                                                 'ESLT.TA', 'ICL.TA',  'LUMI.TA'])


returns = vs.copy()

#port_ret = kit.portfolio_return(rand_w.T, er_w)
# print(port_ret)

# print(vs)

portf_r = []
for w in range(20):
    rets = kit.portfolio_return(vs.iloc[w], er_w)
    #print(np.round(rets, 3))
    portf_r.append(np.round(rets, 3))


# print(portf_r)


returns['Returns'] = portf_r


portf_vol = []
for w in range(20):
    vols = kit.portfolio_vol(vs.iloc[w], cov)
    print(np.round(vols, 3))
    portf_vol.append(np.round(vols, 3))

returns["Volatility"] = portf_vol

print(returns)


returns.to_csv('data/Returns_Volatilities.csv')
