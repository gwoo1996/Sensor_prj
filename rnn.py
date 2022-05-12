import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

df = pd.read_csv('../sprj/data3.csv')


scaler = MinMaxScaler()
df[['sensor_CO','temp_c', 'wind_speed', 'hum_perc']] = scaler.fit_transform(df[['sensor_CO','temp_c', 'wind_speed', 'hum_perc']])

device = torch.device('cuda:0')
print(f'{device} is available')

x = df[['sensor_CO', 'temp_c', 'wind_speed', 'hum_perc']].values
y = df['station_CO'].values

def seq_data(x,y, sequence_length):
    x_seq = []
    y_seq = []
    for i in range(len(x) - sequence_length):
        x_seq.append(x[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])
    
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1,1])

split = 100
sequence_length = 5

x_seq, y_seq = seq_data(x, y, sequence_length)
x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]

print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())

train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

input_size = x_seq.size(2)
num_layers = 2
hidden_size = 8
#hidden_size = 32

class VanillaRNN(nn.Module):

    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(VanillaRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        #self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.Sigmoid())
        self.fc = nn.Linear(hidden_size * sequence_length, 1)
        #self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.ReLU())
        self.relu = nn.ReLU()


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state 설정하기.
        out, _ = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
        out = out.reshape(out.shape[0], -1) # many to many 전략
        out = self.fc(out)
        return out

model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)

criterion = nn.MSELoss()
lr = 1e-3
num_epochs = 300
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_graph = [] # 그래프 그릴 목적인 loss.
loss_test_graph = []
n = len(train_loader)
n_test = len(test_loader)

for epoch in range(num_epochs):
    running_loss = 0.0
    test_loss = 0.0
    for data in train_loader:
        seq, target = data # 배치 데이터.
        out = model(seq)   # 모델에 넣고,
        loss = criterion(out, target) # output 가지고 loss 구하고,
        optimizer.zero_grad() # 
        loss.backward() # loss가 최소가 되게하는 
        optimizer.step() # 가중치 업데이트 해주고,
        running_loss += loss.item() # 한 배치의 loss 더해주고,


    for data_test in test_loader:
        seq_test, target_test = data_test # 배치 데이터.
        out_test = model(seq_test)   # 모델에 넣고,
        loss_test = criterion(out_test, target_test) # output 가지고 loss 구하고,
        test_loss += loss_test.item() # 한 배치의 loss 더해주고,

    loss_test_graph.append(test_loss / n_test)
    loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
    if epoch % 100 == 0:
        print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))

plt.figure(figsize=(20,10))
plt.plot(loss_graph)
plt.show()

plt.figure(figsize=(20,10))
plt.plot(loss_test_graph)
plt.show()

def plotting(train_loader, test_loader, actual):
    with torch.no_grad():
        train_pred = []
        test_pred = []

    for data in train_loader:
        seq, target = data
        out = model(seq)
        #train_pred += out.cpu().numpy().tolist()
        train_pred += out.cpu().tolist()

    for data in test_loader:
        seq, target = data
        out = model(seq)
        #test_pred += out.cpu().numpy().tolist()
        test_pred += out.cpu().tolist()
    

    total = train_pred + test_pred
    plt.figure(figsize=(20,10))
    plt.plot(np.ones(100)*len(train_pred), np.linspace(0,0.1,100), '--', linewidth=0.6)
    plt.plot(actual, '--')
    plt.plot(total, 'b', linewidth=0.6)

    plt.legend(['train boundary', 'actual', 'prediction'])
    plt.show()
    total_series = total
    plot_series = pd.Series(total_series)
    data_for_plot = pd.read_csv('../sprj/data3.csv')
    data_for_plot = pd.concat([data_for_plot, plot_series], axis = 1)
    data_for_plot.to_csv("../sprj/rnn_CO.csv")
    r2_s = r2_score(total, df['station_CO'][sequence_length:].values)
    print(r2_s)



plotting(train_loader, test_loader, df['station_CO'][sequence_length:].values)
