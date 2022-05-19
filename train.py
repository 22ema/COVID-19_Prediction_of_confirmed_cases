# 시계열 데이터는 지도학습 문제로 변환하기 위해서는 예측 대상이 되는 타겟 변수와 예측할 때 사용하는 입력 변수 쌍으로 데이터 가공 필요
# 또한 딥러닝 모델을 안정적으로 학습시키기 위해선 데이터의 스케일을 통일 시키는 작업이 필요하다.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data
from utils.MakeDataset import MakeDataset
from tqdm import tqdm
from model.LSTM import CovidPredictor

def make_Tensor(array):
    return torch.from_numpy(array).float()

def MinMaxScaling(array, min, max):
    return (array-min)/(max-min)

def create_squences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)]
        y = data.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(model, num_epochs, dataloader_dict_, verbose=10, patience=10):
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_hist = []
    val_hist = []
    for t in range(num_epochs):
        epoch_loss = 0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            # for idx, seq in tqdm(dataloader_dict_[phase]): ## 여기 수정중
            #     model.reset_hidden_state() # seq 별 hidden state reset
            #
            #     # train_loss
            #     val_seq = torch.unsqueeze(val_seq, 0)
            #     y_pred = model(seq)
            #     loss = loss_func(y_pred[0].float(), train_labels[idx])
            #
            #     #update weights
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            #
            #     epoch_loss += loss.item()
            # train_hist.append(epoch_loss/len(train_data))
            #
            # if val_data is not None:
            #     with torch.no_grad():

if __name__ == "__main__":
    # make dataset
    time_series = './dataset/time_series_covid19_confirmed_global.csv'
    seq_length = 5
    train_size = int(327 * 0.8)
    batch_size = 10
    model = CovidPredictor()

    time_series_data = pd.read_csv(time_series)
    korea = time_series_data[time_series_data["Country/Region"] == "Korea, South"].iloc[:, 4:].T
    korea.index = pd.to_datetime(korea.index)
    daily_cases = korea.diff().fillna(korea.iloc[0]).astype('int')
    X, y = create_squences(daily_cases, seq_length)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + 33], y[train_size:train_size + 33]
    X_test, y_test = X[train_size + 33:], y[train_size + 33:]

    # data scaling
    # 데이터 범위를 0과 1사이로 변환 시키는 MinMax scaling 진행
    MIN = X_train.min()
    MAX = X_train.max()
    X_train = MinMaxScaling(X_train, MIN, MAX)
    y_train = MinMaxScaling(y_train, MIN, MAX)
    X_val = MinMaxScaling(X_val, MIN, MAX)
    y_val = MinMaxScaling(y_val, MIN, MAX)
    X_test = MinMaxScaling(X_test, MIN, MAX)
    y_test = MinMaxScaling(y_test, MIN, MAX)
    train_dataset = MakeDataset(X_train, y_train)
    val_dataset = MakeDataset(X_val, y_val)
    test_dataset = MakeDataset(X_test, y_test)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}
    train_model(model, 100, dataloader_dict, verbose=10, patience=10)
