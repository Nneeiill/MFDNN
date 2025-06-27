import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import scipy.linalg

# --- Utility Functions ---
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

def Decimal_Scaling(data):
    X_normalized = data / np.max(data)
    return X_normalized

def fit(Xs, Xt):
    cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
    cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
    A_coral = np.dot(
        scipy.linalg.fractional_matrix_power(cov_src, -0.5),
        scipy.linalg.fractional_matrix_power(cov_tar, 0.5)
    )
    Xs_new = np.real(np.dot(Xs, A_coral))
    return Xs_new

# --- Neural Network Definitions ---
class Unit(nn.Module):
    def __init__(self, in_N, out_N):
        super(Unit, self).__init__()
        self.L = nn.Linear(in_N, out_N)
    def forward(self, x):
        x = self.L(x)
        x = F.selu(x)
        return x

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

# --- Model Training Pipeline ---
def train_stacking_mfdnn(X_train, y_train, X_test, y_test, x_train_row, x_test_row):
    torch.manual_seed(1234)
    np.random.seed(1234)
    # Base models
    model1 = RandomForestRegressor(n_estimators=1000, random_state=72)
    model2 = LinearRegression()
    model3 = GradientBoostingRegressor(n_estimators=1000, random_state=72)
    model4 = SVR(C=11.0, epsilon=0.05)
    model1.fit(X_train, y_train.ravel())
    model2.fit(X_train, y_train.ravel())
    model3.fit(X_train, y_train.ravel())
    model4.fit(X_train, y_train.ravel())
    # Stacking features
    pred1 = model1.predict(X_train)
    pred2 = model2.predict(X_train)
    pred3 = model3.predict(X_train)
    pred4 = model4.predict(X_train)
    meta_features = torch.stack((torch.tensor(pred1, dtype=torch.float32),
                                 torch.tensor(pred2, dtype=torch.float32),
                                 torch.tensor(pred3, dtype=torch.float32),
                                 torch.tensor(pred4, dtype=torch.float32)), dim=1)
    fcnn_model = FCNNModel(meta_features.shape[1], 64, 1)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(fcnn_model.parameters(), lr=0.001)
    # Train meta-learner
    for epoch in range(1000):
        optimizer.zero_grad()
        output = fcnn_model(meta_features)
        loss = criterion(output.flatten(), torch.tensor(y_train.reshape(-1), dtype=torch.float32).flatten())
        loss.backward()
        optimizer.step()
    # Test set
    pred1_t = model1.predict(X_test)
    pred2_t = model2.predict(X_test)
    pred3_t = model3.predict(X_test)
    pred4_t = model4.predict(X_test)
    meta_features_test = torch.stack((torch.tensor(pred1_t, dtype=torch.float32),
                                      torch.tensor(pred2_t, dtype=torch.float32),
                                      torch.tensor(pred3_t, dtype=torch.float32),
                                      torch.tensor(pred4_t, dtype=torch.float32)), dim=1)
    final_pred_tensor = fcnn_model(meta_features_test)
    mape = mean_absolute_percentage_error(y_test, final_pred_tensor.detach().numpy())
    return final_pred_tensor, mape
