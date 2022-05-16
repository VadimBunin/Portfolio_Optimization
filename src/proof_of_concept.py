import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

df_bezeq = pd.read_csv("data/BEZQ.TA.csv", parse_dates=True, index_col=0)
df_bezeq = df_bezeq[['Close']]
df_bezeq.plot(figsize=(10, 6))
#plt.title('Leumi Stock History')
# plt.show()
scaler = MinMaxScaler(feature_range=(-1, 1))
df_bezeq = scaler.fit_transform(df_bezeq.values.reshape(-1, 1))

# Create Seqs and Labels


def load_data(stock, window_size):
    data_raw = stock
    data = []

    for index in range(len(data_raw) - window_size):
        data.append(data_raw[index: index + window_size])

    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]

    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


ws = 63
x_train, y_train, x_test, y_test = load_data(df_bezeq, ws)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

# Convert to Ternsors
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

# LSTM Model

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

num_epochs = 250
hist = np.zeros(num_epochs)


for t in range(num_epochs):

    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train)
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
y_test_pred = model(x_test)

# Convert to numpy

y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

# MSE, RMSE

MAE_tr = mean_absolute_error(y_train[:, 0], y_train_pred[:, 0])
print(MAE_tr)
MAE = mean_absolute_error(y_test[:, 0], y_test_pred[:, 0])
print('Test Score: %.2f MSE' % (MAE))
RMSE = np.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
print('Test Score: %.2f RMSE' % (RMSE))

df_bezeq = pd.read_csv("data/BEZQ.TA.csv", parse_dates=True, index_col=0)
df_bezeq = df_bezeq[['Close']]

# Visualising the results
figure, axes = plt.subplots(figsize=(8, 3))
axes.xaxis_date()

axes.plot(df_bezeq[len(df_bezeq)-len(y_test):].index, y_test,
          color='red', label='BEZEQ History Stock Price')
axes.plot(df_bezeq[len(df_bezeq)-len(y_test):].index, y_test_pred,
          color='blue', label='Predicted BEZEQ Stock Price')
# axes.xticks(np.arange(0,394,50))
plt.title('BEZEQ Stock Price Prediction')
plt.ylabel('BEZEQ Stock Price')
plt.legend()
plt.show()
