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
df[['sensor_SO2','temp_c', 'wind_speed', 'hum_perc']] = scaler.fit_transform(df[['sensor_SO2','temp_c', 'wind_speed', 'hum_perc']])

device = torch.device('cuda:0')
print(f'{device} is available')

x = df[['sensor_SO2', 'temp_c', 'wind_speed', 'hum_perc']].values
y = df['station_SO2'].values

def seq_data(x,y, sequence_length):
    x_seq = []
    y_seq = []
    for i in range(len(x) - sequence_length):
        x_seq.append(x[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])
    
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1,1])

split = 100
sequence_length = 1
# fixed for simple dnn

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
hidden_size = 16
dim_out = 1

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.layer_hidden1 = nn.Linear(dim_hidden, 8)
        self.layer_fc2 = nn.Linear(8, 12)

        self.layer_fc = nn.Linear(dim_hidden, dim_out)
        
        self.layer_test1 = nn.Linear(dim_in, 8)
        self.layer_test2 = nn.Linear(8, 12)
        self.layer_test3 = nn.Linear(12, 24)
        self.layer_test4 = nn.Linear(24, 12)
        self.layer_test5 = nn.Linear(12, 4)
        self.layer_testfc = nn.Linear(4, 1)

        self.layer_simple = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        # x = self.layer_input(x)
        # x = self.relu(x)
    
        # x = self.layer_hidden1(x)
        # x = self.relu(x)  

        # x = self.layer_fc2(x)

        x = self.layer_test1(x)
        x = self.relu(x)
        x = self.layer_test2(x)
        x = self.relu(x)
        x = self.layer_test3(x)
        x = self.relu(x)
        x = self.layer_test4(x)
        x = self.relu(x)
        x = self.layer_test5(x)
        x = self.relu(x)
        x = self.layer_testfc(x)

        return x



model = MLP(dim_in = input_size, dim_hidden = hidden_size, dim_out = dim_out).to(device)

criterion = nn.MSELoss()
lr = 1e-3
num_epochs = 300
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)

for epoch in range(num_epochs):
  running_loss = 0.0

  for data in train_loader:

    seq, target = data # 배치 데이터.
    out = model(seq)   # 모델에 넣고,
    loss = criterion(out, target) # output 가지고 loss 구하고,
    optimizer.zero_grad() # 
    loss.backward() # loss가 최소가 되게하는 
    optimizer.step() # 가중치 업데이트 해주고,
    running_loss += loss.item() # 한 배치의 loss 더해주고,

  loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
  if epoch % 100 == 0:
    print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))

plt.figure(figsize=(20,10))
plt.plot(loss_graph)
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
    
    data_for_plot = pd.read_csv('../sprj/data3.csv')

    total = train_pred + test_pred
    total = np.array(total)
    total = total.reshape(-1,)
    print(total.shape)

    

    plt.figure(figsize=(20,10))
    plt.plot(np.ones(100)*len(train_pred), np.linspace(0,0.1,100), '--', linewidth=0.6)
    plt.plot(actual, '--')
    plt.plot(total, 'b', linewidth=0.6)

    plt.legend(['train boundary', 'actual', 'prediction'])
    plt.show()

    #np.swapaxes(total, 0, 1)
    #total_series = np.insert(total, 0, 3.141592)
    total_series = total
    plot_series = pd.Series(total_series)
    data_for_plot = pd.concat([data_for_plot, plot_series], axis = 1)
    data_for_plot.to_csv("../sprj/dnn_SO2.csv")
    r2_s = r2_score(total, df['station_SO2'][sequence_length:].values)
    print(r2_s)
    plt.scatter(total, df['station_SO2'][sequence_length:].values)
    plt.show()


plotting(train_loader, test_loader, df['station_SO2'][sequence_length:].values)


