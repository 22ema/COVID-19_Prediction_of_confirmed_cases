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

def MAE(true, pred):
    return np.mean(np.abs(true-pred))

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
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_hist = list()
    val_hist = list()
    for t in range(num_epochs):
        print("")
        print('---------------------')
        print("epoch :{}".format(t+1))

        for phase in ['train', 'val']:
            epoch_loss = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for inputs, labels in tqdm(dataloader_dict_[phase]): ## 여기 수정중
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    model.reset_hidden_state() # seq 별 hidden state reset
                    # train_loss
                    pred = model(inputs)
                    loss = criterion(pred, labels)
                    #update weights

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                train_hist.append(epoch_loss/len(dataloader_dict_[phase].dataset))
            else:
                val_hist.append(epoch_loss/len(dataloader_dict_[phase].dataset))
        if (t+1) % verbose==0:
            print("")
            print(f'Epoch {t+1} train loss: {train_hist[-1]} val loss: {val_hist[-1]}')
    return model, train_hist, val_hist

def inferencing(model, test_dataset):
    model = model.eval()
    with torch.no_grad():
        preds = list()
        for inputs, idx in test_dataset:
            model.reset_hidden_state()
            y_test_pred = model(inputs)
            pred = torch.flatten(y_test_pred).item()
            preds.append(pred)
    return preds

if __name__ == "__main__":
    # make dataset
    time_series = './dataset/time_series_covid19_confirmed_global.csv'
    seq_length = 5
    train_size = int(327 * 0.8)
    batch_size = 10

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
    X_train = make_Tensor(X_train)
    y_train = make_Tensor(y_train)
    X_val = make_Tensor(X_val)
    y_val = make_Tensor(y_val)
    X_test = make_Tensor(X_test)
    y_test = make_Tensor(y_test)
    train_dataset = MakeDataset(X_train, y_train)
    val_dataset = MakeDataset(X_val, y_val)
    test_dataset = MakeDataset(X_test, y_test)
    train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}
    model = CovidPredictor(n_features=1, n_hidden=4, seq_len=seq_length, n_layers=1)
    model, train_hist, val_hist = train_model(model, 100, dataloader_dict, verbose=10, patience=10)
    plt.plot(train_hist, label="Training loss")
    plt.plot(val_hist, label="Val loss")
    plt.legend()
    plt.show()
    preds = inferencing(model, test_dataloader)
    MAE_result = MAE(np.array(y_test)*MAX, np.array(preds)*MAX)
    print(MAE_result)
    plt.plot(daily_cases.index[-len(y_test):], np.array(y_test) * MAX, label='True')
    plt.plot(daily_cases.index[-len(preds):], np.array(preds) * MAX, label='Pred')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
