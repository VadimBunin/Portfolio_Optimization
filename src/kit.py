
import pandas as pd
import numpy as np


def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())


def find_winner(symbol):
    for col in symbol:
        acc = ((1+six_month[col]).prod() - 1)
        if acc > ((1+six_month['TA35.TA']+.018).prod() - 1):
            print('Winner', col, " ", round(acc, 3))

        elif acc == ((1+six_month['TA35.TA']).prod() - 1):
            print('Benchmark', col, "", round(acc, 3))

        else:
            print('Looser', col, " ", round(acc, 3))
