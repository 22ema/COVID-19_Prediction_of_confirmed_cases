import pandas as pd
import numpy as np
from torch.utils import data

class MakeDataset(data.Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]
