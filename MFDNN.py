import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    percentage_errors = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(percentage_errors) * 100
    return mape

def stress_cal(Ys, E, ep, n):
    stress = Ys * (((E / Ys) * ep) ** n)
    return stress

def min_max_normalization2(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    denominator = max_vals - min_vals
    denominator[denominator == 0] = 1e-8
    norm_data = (data - min_vals) / denominator
    return norm_data

class Unit(nn.Module):
    def __init__(self, in_N, out_N):
        super(Unit, self).__init__()
        self.L = nn.Linear(in_N, out_N)
    def forward(self, x):
        x1 = self.L(x)
        x2 = F.selu(x1)
        return x2

class NN1(nn.Module):
    def __init__(self, in_N, width, depth, out_N):
        super(NN1, self).__init__()
        self.stack = nn.ModuleList()
        self.stack.append(Unit(in_N, width))
        for _ in range(depth):
            self.stack.append(Unit(width, width))
        self.stack.append(nn.Linear(width, out_N))
    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x

class NN2(nn.Module):
    def __init__(self, in_N, width, depth, out_N):
        super(NN2, self).__init__()
        self.stack = nn.ModuleList()
        self.stack.append(nn.Linear(in_N, width))
        for _ in range(depth):
            self.stack.append(nn.Linear(width, width))
        self.stack.append(nn.Linear(width, out_N))
    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

class FCNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

torch.manual_seed(1234)
np.random.seed(1234)

# data read
# df = pd.read_excel('Strain-Stress.xlsx', sheet_name=0)
# dfs = ...



# X_train = ...
# y_train = ...
# X_test = ...
# y_test = ...
# x_train_row = ...
# x_test_row = ...


alpha3 = torch.tensor([0.0], requires_grad=True)
alpha4 = torch.tensor([0.0], requires_grad=True)

model_h_s = NN1(40, 128, 5, 35)
model_h_s.apply(weights_init)
model4_1 = NN2(75, 64, 3, 35)
model4_1.apply(weights_init)

optimizer3 = optim.Adam([
    {'params': model_h_s.parameters(), 'weight_decay': 0.01},
    {'params': model4_1.parameters(), 'weight_decay': 0.01},
    {'params': [alpha3],'lr': 1e-3},
    {'params': [alpha4],'lr': 1e-3}], lr=1e-3)

model1 = RandomForestRegressor(n_estimators=1000, random_state=72)
model2 = LinearRegression()
model3 = GradientBoostingRegressor(n_estimators=1000, random_state=72)
model4 = SVR(C=11.0, epsilon=0.05)

model1.fit(X_train, y_train.ravel())
model2.fit(X_train, y_train.ravel())
model3.fit(X_train, y_train.ravel())
model4.fit(X_train, y_train.ravel())

pred1 = model1.predict(X_train)
pred2 = model2.predict(X_train)
pred3 = model3.predict(X_train)
pred4 = model4.predict(X_train)

meta_features = torch.stack((torch.tensor(pred1, dtype=torch.float32),
                             torch.tensor(pred2, dtype=torch.float32),
                             torch.tensor(pred3, dtype=torch.float32),
                             torch.tensor(pred4, dtype=torch.float32)), dim=1)

input_size = meta_features.shape[1]
hidden_size = 64
output_size = 1
fcnn_model = FCNNModel(input_size, hidden_size, output_size)

criterion = nn.L1Loss()
optimizer = optim.Adam(fcnn_model.parameters(), lr=0.001)

for epoch in range(10000):
    optimizer.zero_grad()
    output = fcnn_model(meta_features)
    loss = criterion(output.flatten(), torch.tensor(y_train.reshape(-1), dtype=torch.float32).flatten())
    loss.backward()
    optimizer.step()

