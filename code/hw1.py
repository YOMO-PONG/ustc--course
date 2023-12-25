import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

# define the neural network
class Net(nn.Module):
    def __init__(self, activation, layers):
        super(Net, self).__init__()
        self.activation = activation
        self.num_layers = len(layers)-1
        self.fctions = nn.ModuleList()
        # 建立线性层
        for i in range(self.num_layers):
            self.fctions.append(nn.Linear(layers[i], layers[i+1]))
        # 加入激活函数形成整个前向传播的计算框架
    def forward(self, x):
        for i in range(self.num_layers-1):
            x = self.activation(self.fctions[i](x))
        x = self.fctions[-1](x)
        return x
    
# Define a lambda function that returns the learning rate multiplier
# based on the current iteration/epoch
def lr_lambda(current_step):
    warmup_steps = 1000
    if current_step < warmup_steps:
        return float(current_step/warmup_steps)
    else:
        return 1.0

def Training(traindata , valdata, layers, activation, lr):
    # create the network, optimizer, and loss function
    net = Net(activation = activation, layers = layers)
    optimizer = optim.Adam(net.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-08, 
                           weight_decay=0, amsgrad=True)
    loss_func = nn.MSELoss()
    # Create a learning rate scheduler that uses the lambda function
    scheduler = LambdaLR(optimizer, lr_lambda)
    # train the network
    x_train, y_train = traindata
    x_val, y_val = valdata
    loss_train = []
    loss_val = []
    #设置超参数epoch

    for i in range(1000):
        optimizer.zero_grad()
        y_pred = net(x_train)
        loss_t = loss_func(y_pred, y_train)
        loss_t.backward()
        # 更新权重
        optimizer.step()
        # 更新学习率
        scheduler.step()
        loss_train.append(loss_t.item())
        y_vpred = net(x_val)
        loss_v = loss_func(y_vpred, y_val)
        loss_val.append(loss_v.item())
        
        t=i+1
        if  t% 100 == 0:
            print(f"\t\t Epoch: {t}, Loss: {loss_v.item()}")
        if loss_v.item()<1e-6:
            print(f"\t\t Epoch: {t}, Loss: {loss_v.item()}")
            break 
    loss_curve = [loss_train, loss_val]
    return net, loss_curve

def date_init(n):
    def func(x):
        return np.sin(x)

    a = 0
    b = 2 * np.pi
    t = int(0.8 * n)
    v = int(0.9 * n)
    # create the data
    x = np.random.uniform(a, b, size=(n,1))
    y = func(x)

    x_train = torch.from_numpy(x[0 : t]).float()
    x_val = torch.from_numpy(x[t : v]).float()
    x_test = torch.from_numpy(x[v : n]).float()

    y_train = torch.from_numpy(y[0 : t]).float()
    y_val = torch.from_numpy(y[t : v]).float()
    y_test = torch.from_numpy(y[v : n]).float()
    return x_train,x_val,x_test,y_train,y_val,y_test
"""
y_tpred = net(x_test)
mse = nn.MSELoss()(y_tpred, y_test)
print(f"测试集mse: {mse.item()}")
plt.figure(figsize=(8,4),dpi=300)
plt.scatter(x_test, y_test, label='Truth', s=1)
plt.scatter(x_test, y_tpred.detach().numpy(), label='Pred', s=1)
plt.legend()
plt.show()
"""
def main():
    x_train,x_val,x_test,y_train,y_val,y_test = date_init(10000)
    #activation = torch.sigmoid
    depth = 10
    width = 50
    lr = 0.005
    activation = torch.relu
    print('验证集mse:')
    layers = [1]
    for i in range(depth):
        layers.append(width)
    layers.append(1)
    net, loss_curve = Training((x_train, y_train), (x_val, y_val), layers, activation, lr)
    t = len(loss_curve[0])
    plt.figure(figsize=(8, 4), dpi=300)
    plt.title('loss_curves')
    plt.plot(range(t), loss_curve[0])
    plt.plot(range(t), loss_curve[1])
    plt.legend(['train', 'val'])
    save_path = "/Users/pongyomo/Documents/ustc-course/DL/exp1-pc"
    file_name = f"Depth_{depth}_Width_{width}.png"
    #plt.savefig(f"{save_path}/{file_name}")
    plt.show()
    y_tpred = net(x_test)
    mse = nn.MSELoss()(y_tpred, y_test)
    print(f"测试集mse: {mse.item()}")
    plt.figure(figsize=(8, 4), dpi=300)
    plt.scatter(x_test, y_test, label='Truth', s=1)
    plt.scatter(x_test, y_tpred.detach().numpy(), label='Pred', s=1)
    plt.legend()
    plt.show()
if __name__ == "__main__":
    # print(torch.cuda.is_available())
    main()
