from doctest import DocFileCase
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

start = "2014-01-01"
end = "2022-03-31"
df = yf.download('ICL.TA', start, end)
df.to_csv("data/'ICL.TA.csv")
df = pd.read_csv("data/'ICL.TA.csv", header=0, index_col=0, parse_dates=[0])
print(df.head())

close = df.Close.copy()
returns = close.pct_change().copy()
returns.dropna(inplace=True)
df = returns.resample('M').apply(kit.compound).to_period('M')
print(df[-1])
df = df[:-1]

# print(stock)
scaler = MinMaxScaler(feature_range=(-1, 1))
df = scaler.fit_transform(df.values.reshape(-1, 1))


def load_data(stock, window_size):
    data_raw = stock
    data = []

    for index in range(len(data_raw) - window_size):
        data.append(data_raw[index: index + window_size])

    data = np.array(data)

    x = data[:, :-1, :]

    y = data[:, -1, :]

    return [x, y]


ws = 4
x, y = load_data(df, ws)
print('x.shape = ', x.shape)
print('y.shape = ', y.shape)

x = torch.from_numpy(x).type(torch.Tensor)

y = torch.from_numpy(y).type(torch.Tensor)

input_dim = 1
hidden_dim = 128
num_layers = 2
output_dim = 1


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,  output_dim, drop_prob=0.0):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


torch.manual_seed(22)

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim,
             output_dim=output_dim, num_layers=num_layers)

criterion = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 901
hist = np.zeros(num_epochs)


for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    # model.hidden = model.init_hidden()

    # Forward pass
    y_train_pred = model(x)

    loss = criterion(y_train_pred, y)
    if t % 100 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()
y_pred = model(x)

y_pred = y_pred.detach().numpy()

y_pred = np.expand_dims(y_pred, axis=0)

window_size = 4
future = 1
L = len(y)

preds = y_pred[-window_size:].tolist()

model.eval()
for i in range(future):
    seq = torch.FloatTensor(y_pred[-3:])
    with torch.no_grad():

        preds.append(model(seq).item())

ICL = scaler.inverse_transform(np.array(preds[-1]).reshape(-1, 1))
ICL = ICL.astype(float)

print(ICL)

TA_m = pd.read_csv('data/Momentum.csv', index_col=0)

TA_m['ICL.TA'] = ICL

print(TA_m.T)

TA_m.to_csv('data/Momentum.csv')
