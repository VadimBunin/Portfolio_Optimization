import pandas as pd
import numpy as np
import kit

df = pd.read_csv('data/ta_35_monthly_returns.csv',
                 index_col=0, parse_dates=True)

# returns for the last 6 month

df = df.to_period('M')
# print(df)
six_month = df[-7:-1]
# print(six_month)


winner = []
for col in six_month:

    acc = ((1+six_month[col]).prod() - 1)

    if acc > ((1+six_month['TA35.TA']+.02).prod() - 1):

        winner.append(col)

        print('Winner', col, " ", round(acc, 3))

    elif acc == ((1+six_month['TA35.TA']).prod() - 1):

        print('Benchmark', col, "", round(acc, 3))

    else:

        print('Looser', col, " ", round(acc, 3))

print(winner)


# df[winner].to_csv("data/winner.csv")
