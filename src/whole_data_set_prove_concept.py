import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("data/BEZQ.TA.csv", parse_dates=True, index_col=0)
df = df[['Close']]
print(df.iloc[-1])
df = df.iloc[:-1]

scaler = MinMaxScaler(feature_range=(-1, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))


def inpit_data(stock, ws):
    stock = stock.values
    data = []

    for i in range(len(stock) - ws):
        data.append(stock[i:i+ws])
    data = np.array(data)

    x = data[:, :-1, :]
    y = data[:, -1, :]

    return [x, y]


x, y = inpit_data(df, 63)


print('The shape of  x is ', x.shape)
print('The shape of y is ', y.shape)

x = torch.from_numpy(x).type(torch.Tensor)
y = torch.from_numpy(y).type(torch.Tensor)

input_dim = 1
hidden_dim = 64
num_layers = 2
output_dim = 1


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0, c0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_dim),
                  torch.zeros(self.num_layers, x.size(0), self.hidden_dim))

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])

        return out


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim,
             output_dim=output_dim, num_layers=num_layers)

criterion = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model

num_epochs = 100
hist = np.zeros(num_epochs)


for t in range(num_epochs):

    y_train_pred = model(x)

    loss = criterion(y_train_pred, y)
    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

#plt.plot(hist, label="Training loss")
# plt.legend()
# plt.show()

# Predictions
y_pred = model(x)

y_pred = y_pred.detach().numpy()

y_pred = np.expand_dims(y_pred, axis=0)

window_size = 63
future = 1
L = len(y)

preds = y_pred[-window_size:].tolist()

model.eval()
for i in range(future):
    seq = torch.FloatTensor(y_pred[-window_size:])
    with torch.no_grad():

        preds.append(model(seq).item())

BEZQ = scaler.inverse_transform(np.array(preds[-1]).reshape(-1, 1))
BEZQ = BEZQ.astype(float)

print(BEZQ)
