import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim, autograd
from torch.nn import functional as F
import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors

#read data
df = pd.read_excel('C:\\Users\\yangz\\Desktop\\data download\\Punch_Steel_Al_20230821.xlsx',sheet_name='K-n_Stress-Strain' )
first_row = df.iloc[0].values.reshape(1,-1)
dfs = pd.read_excel('C:\\Users\\yangz\\Desktop\\data download\\Low_input_new.xlsx',sheet_name=0 )
dfs = dfs.values
X_low = np.loadtxt('C:\\Users\\yangz\\Desktop\\code\\highdata\\low_data_input_C.text', delimiter=',')
Y_low = np.loadtxt('C:\\Users\\yangz\\Desktop\\code\\highdata\\low_data_output.text', delimiter=',')
X_high = np.loadtxt('C:\\Users\\yangz\\Desktop\\code\\highdata\\exp_in_2.text',delimiter=',')
Y_high =np.loadtxt('C:\\Users\\yangz\\Desktop\\code\\highdata\\exp_out_2.text',delimiter=',')



torch.manual_seed(1234)
np.random.seed(1234)

class Unit(nn.Module):

    def __init__(self, in_N, out_N):
        super(Unit, self).__init__()
        self.in_N = in_N
        self.out_N = out_N
        self.L = nn.Linear(in_N, out_N)

    def forward(self, x):
        x1 = self.L(x)
        x2 = F.selu(x1)
        return x2


class NN1(nn.Module):

    def __init__(self, in_N, width, depth, out_N):
        super(NN1, self).__init__()
        self.width = width
        self.in_N = in_N
        self.out_N = out_N
        self.stack = nn.ModuleList()

        self.stack.append(Unit(in_N, width))

        for i in range(depth):
            self.stack.append(Unit(width, width))

        self.stack.append(nn.Linear(width, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x


class NN2(nn.Module):
    def __init__(self, in_N, width, depth, out_N):
        super(NN2, self).__init__()
        self.in_N = in_N
        self.width = width
        self.depth = depth
        self.out_N = out_N

        self.stack = nn.ModuleList()

        self.stack.append(nn.Linear(in_N, width))

        for i in range(depth):
            self.stack.append(nn.Linear(width, width))

        self.stack.append(nn.Linear(width, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def mean_absolute_percentage_error(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 防止除以零的情况
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # 计算百分比误差
    percentage_errors = np.abs((y_true - y_pred) / y_true)
    
    # 计算MAPE
    mape = np.mean(percentage_errors) * 100
    return mape

def stress_cal(Ys, E, ep, n):
    stress =  Ys * (((E / Ys) * ep) ** n)
    return stress
# preprocessing
# import numpy as np
# import scipy.io
# import scipy.linalg
# import sklearn.metrics
# import sklearn.neighbors
def min_max_normalization2(data):
    min_vals = np.min(data, axis=0)  
    max_vals = np.max(data, axis=0)  
    denominator = max_vals - min_vals
    denominator[denominator == 0] = 1e-8  # 防止分母为零
    norm_data = (data - min_vals) / denominator
    return norm_data

def Decimal_Scaling(data):
    X_normalized = data / np.max(data)
    return X_normalized

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def fit(Xs, Xt):
    '''
    Perform CORAL on the source domain features
    :param Xs: ns * n_feature, source feature
    :param Xt: nt * n_feature, target feature
    :return: New source domain features
    '''
    cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
    cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
    A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                        scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
    np.linalg.multi_dot([Xs, scipy.linalg.fractional_matrix_power(cov_src, -0.5), scipy.linalg.fractional_matrix_power(cov_tar, 0.5)])
    Xs_new = np.real(np.dot(Xs, A_coral))
    return Xs_new

def main():
    #low-fidelity data
    x_hi = dfs[[104,130],:]
    x_hi = min_max_normalization2(x_hi)
    x_lo = dfs[[0,26,52,78],:]
    x_lo = min_max_normalization2(x_lo)

    #high-fidelity data
    y_hi_star = Y_low[[104,130],1].reshape(-1,1)
    y_hi_max = np.max(y_hi_star)
    y_hi_star = Decimal_Scaling(y_hi_star)
    y_lo_star = Y_low[[0,26,52,78],1].reshape(-1,1)
    y_lo_max = np.max(y_lo_star)
    y_lo_star = Decimal_Scaling(y_lo_star)

    # #test dataa
    x = dfs[[152],:].reshape(1,-1)
    x = min_max_normalization2(x)
    y_hi = Y_low[[152],1].reshape(-1,1)
    y_lo_max = np.max(y_lo_star)
    y_hi = Decimal_Scaling(y_hi)

    x_train = np.concatenate((x_lo, x_hi), axis=0)
    

    #step 2 using low+highfidelity data to predict

    model_h = NN1(5, 20, 4, 1)
    model_h.apply(weights_init)
    optimizer = optim.Adam(model_h.parameters(), lr=1e-3)
    loss_value = 1
    x_lo_r = torch.from_numpy(x_lo).float()
    x_lo_r.requires_grad_() #在这段代码中，x_lo_r 被设置为需要梯度，因为在训练低保真度模型时，我们需要计算损失函数关于输入的梯度，并使用它来计算梯度损失项
    it = 0
    nIter1 = 3000
    while loss_value > 1e-3 and it < nIter1:
        pred_h = model_h(x_lo_r)
        # print(pred_h)
        grads = autograd.grad(outputs=pred_h, inputs=x_lo_r,
                            grad_outputs=torch.ones_like(pred_h), #grad_outputs：可选参数，用于指定输出梯度的形状和数据类型。这里使用 torch.ones_like(pred_h)，表示输出梯度的形状与 pred_h 相同，并且所有元素的值都为1。
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        # loss = torch.mean(torch.square(pred_h - torch.from_numpy(y_lo_star).float())) + \
        #        torch.mean(torch.sum(torch.square(grads - torch.from_numpy(y_lo_star_prime).float()), 1, keepdim=True))
        loss = torch.mean(torch.square(pred_h - torch.from_numpy(y_lo_star).float()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if it % 100 == 0:
            print('It:', it, 'Loss', loss.item())
        it = it + 1



    alpha1 = torch.tensor([0.0],requires_grad=True)
    alpha2 = torch.tensor([0.0], requires_grad=True)
    model3 = NN1(6, 32, 4, 1)
    model4 = NN2(6, 32, 3, 1)
    model3.apply(weights_init)
    model4.apply(weights_init)
    optimizer2 = optim.Adam([{'params': model3.parameters(), 'weight_decay': 0.01},
                            {'params': model_h.parameters(), 'weight_decay': 0.01},
                            {'params': model4.parameters(), 'weight_decay': 0.01},
                            {'params': alpha1,'lr': 1e-3},
                            {'params': alpha2,'lr': 1e-3}], lr=1e-3)

    nIter2 = 3000
    x_lo_r = torch.from_numpy(x_lo).float()
    x_lo_r.requires_grad_()
    loss2_value = 1
    it = 0
    # first_row = first_row.reshape(-1,1)
    # x_lo_r.requires_grad_()
    while  it < nIter2:
        
        pred_h = model_h(x_lo_r)
        grads = autograd.grad(outputs=pred_h, inputs=x_lo_r,
                            grad_outputs=torch.ones_like(pred_h),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        loss3 = torch.mean(torch.square(pred_h - torch.from_numpy(y_lo_star).float())) 
            #    torch.mean(torch.sum(torch.square(grads - torch.from_numpy(y_lo_star_prime).float()), 1, keepdim=True))

        pred_2h = model_h(torch.from_numpy(x_hi).float())
        pred_2 =  pred_2h + 0.1*(torch.tanh(alpha1) * model3(torch.cat((torch.from_numpy(x_hi).float(), pred_2h), 1)) + \
                torch.tanh(alpha2) * model4(torch.cat((torch.from_numpy(x_hi).float(), pred_2h), 1)))
        loss2 = torch.mean(torch.square(pred_2 - torch.from_numpy(y_hi_star).float())) + loss3
        loss2_value = loss2.item()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        if it % 100 == 0:
            print('It:', it, 'Loss:', loss2.item(),'a1:',alpha1,'a2',alpha2)
        it = it + 1


    predict_model_h = model_h(torch.from_numpy(x_train).float())
    predict_ys = predict_model_h  + 0.1*(torch.tanh(alpha1) * model3(torch.cat((torch.from_numpy(x_train).float(), predict_model_h), 1)) + \
                torch.tanh(alpha2) * model4(torch.cat((torch.from_numpy(x_train).float(), predict_model_h), 1)))

    alpha3 = torch.tensor([0.0],requires_grad=True)
    alpha4 = torch.tensor([0.0],requires_grad=True)
    model_h_s = NN1(40, 128, 5, 35)
    model_h_s.apply(weights_init)
    optimizer_s = optim.Adam(model_h_s.parameters(), lr=1e-3)
    model3_1 = NN1(75, 128, 5, 35)
    model4_1 = NN1(75, 64, 3, 35)

    model3_1.apply(weights_init)
    model4_1.apply(weights_init)


    optimizer3 = optim.Adam([{'params': model3_1.parameters(), 'weight_decay': 0.01},
                            {'params': model_h_s.parameters(), 'weight_decay': 0.01},
                            {'params': model4_1.parameters(), 'weight_decay': 0.01},
                            {'params': [alpha3],'lr': 1e-3},
                            {'params': [alpha4],'lr': 1e-3}], lr=1e-3)

    pre_n = predict_ys.detach().numpy()
    # first_row = first_row.reshape(1,-1)

    for i in range(len(pre_n)):
        print('i=',i)
        # pre_n = pred_2.detach().numpy()
        Ys_p = pre_n[i]*127.17
        E = 70000
        n = 0.068
        #low-fidelity 
        lo_strain = np.linspace(0, 1, 35).reshape(1,-1).astype(np.float32)
        lo_stress = stress_cal(Ys_p,E,lo_strain,n).astype(np.float32)
        lo_strain = torch.from_numpy(lo_strain).float()
        input_strain_low = torch.cat((torch.from_numpy(x_train[i,:].reshape(1,-1)).float(), lo_strain), dim=1)
    
        #high fidelity 
        hi_stress = first_row[0,42:].astype(np.float32).reshape(1,-1)
        hi_stress = torch.from_numpy(hi_stress).float()
        hi_strain = first_row[0,7:42,].astype(np.float32).reshape(1,-1)
        hi_strain = torch.from_numpy(hi_strain).float()
        input_hi_strain = torch.cat((torch.from_numpy(x_train[i,:].reshape(1,-1)).float(), hi_strain), dim=1)





        nIter2_s = 2000
        # lo_r_strain = torch.from_numpy(input_strain_low).float()
        input_strain_low.requires_grad_()
        loss2_value = 1
        it_s = 0
        # x_lo_r.requires_grad_()
        while it_s < nIter2_s:
            pred_h_s = model_h_s(input_strain_low)
            grads = autograd.grad(outputs=pred_h_s, inputs=input_strain_low,
                                grad_outputs=torch.ones_like(pred_h_s),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
            loss_s = torch.mean(torch.square(pred_h_s - torch.from_numpy(lo_stress).float())) + 0.01 * sum(torch.norm(param)**2 for param in model_h_s.parameters())

            pred_h_s_2 = model_h_s(input_hi_strain.float())
            pred_2_s =  pred_h_s_2 + 0.1*(torch.tanh(alpha3) * model3_1(torch.cat((input_hi_strain.float(), pred_h_s_2), 1)) + \
            torch.tanh(alpha4) * model4_1(torch.cat((input_hi_strain.float(), pred_h_s_2), 1)))
            
            loss2_s = torch.mean(torch.square(pred_2_s - hi_stress.float())) + loss_s
            # loss_list.append(loss2_s.item())
        
            optimizer3.zero_grad()
            loss2_s.backward()
            optimizer3.step()
            loss2_value = loss2_s.item()
            # loss2_s = loss2_s.item()
            if it_s % 100 == 0:
                print('it_s:', it_s, 'Loss', loss2_s.item())
            it_s = it_s+1

    xx_lo = model_h(torch.from_numpy(x).float())
    xx_high = xx_lo + 0.1*(torch.tanh(alpha1)* model3(torch.cat((torch.from_numpy(x).float(), xx_lo), 1)) +\
            torch.tanh(alpha2) * model4(torch.cat((torch.from_numpy(x).float(), xx_lo), 1)))
    strain = np.linspace(0, 1, 35).reshape(1,-1)
    pre_Ys = xx_high[0].detach().numpy()
    pre_Ys = pre_Ys * 127.17
    # y_lo_strain = y_lo_strain.detach().numpy()
    pre_Stress = stress_cal(pre_Ys, E, strain, n).astype(np.float32)

    x_hi_s = first_row[0, 7:42].astype(np.float32).reshape(1,-1)
    y_hi_s = first_row[0, 42:].astype(np.float32).reshape(1,-1)

    y_hi_strain = torch.cat((torch.from_numpy(x[0,:].reshape(1,-1)).float(), hi_strain), dim=1)
    pred_h_s_2_t  = model_h_s(y_hi_strain)
    pred_2_s_t =  pred_h_s_2_t + 0.1*(torch.tanh(alpha3) * model3_1(torch.cat((y_hi_strain, pred_h_s_2_t), 1)) + \
    torch.tanh(alpha4) * model4_1(torch.cat((y_hi_strain, pred_h_s_2_t), 1)))
    y_hi = 127.17
    pre_Stress_t = stress_cal(y_hi, E, strain, n).astype(np.float32)
    pred_2_s_t = pred_2_s_t.detach().numpy()
    pred_2_s_t = pred_2_s_t.reshape(1,-1)
    strain = strain.reshape(1,-1)
    pre_Stress = pre_Stress.reshape(1,-1)
    pre_Stress_t = torch.from_numpy(pre_Stress_t).float()
    pre_Stress_t = pre_Stress_t.reshape(1,-1)
    fig4, ax4 = plt.subplots()
    ax4.plot(strain.flatten(),pre_Stress.flatten(), label='calculation by predicted Ys', color='green')
    ax4.plot(x_hi_s.flatten(),y_hi_s.flatten(), label='$Exact$', color='black')
    ax4.plot(strain.flatten(),pre_Stress_t.flatten(), label='$calculation by true Ys$', color='red')
    ax4.plot(strain.flatten(),pred_2_s_t.flatten(), label='$MFDNN Curev$', color='blue')
    plt.legend()
    plt.show()

    print('pre_Ys:',pre_Ys)
    print('pre_Stress:',pre_Stress)
    print('true:',y_hi)
    print('pred:',xx_high*127.17)
    print(alpha1)
    print(alpha2)

if __name__ == '__main__':
    main()