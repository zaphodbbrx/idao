import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


class SateliteDataset(Dataset):

    def __init__(self, csv_path: str, feature_columns: list, target_columns: list, seq_len=5):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.len = file_len(csv_path)
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.df = pd.read_csv(csv_path, chunksize=1)
        df_t = pd.read_csv(csv_path, nrows=100000)
        self.scaler_x = MinMaxScaler().fit(df_t[feature_columns].values)
        self.scaler_y = MinMaxScaler().fit(df_t[target_columns].values)

    def __getitem__(self, item):
        features = []
        for i in range(self.seq_len):
            d = next(self.df)
            features.append(d[self.feature_columns].values)
            targets = d[self.target_columns].values
        features = self.scaler_x.transform(np.vstack(features).astype(np.float32))
        targets = np.squeeze(self.scaler_y.transform(targets.astype(np.float32)))
        return features, targets

    def __len__(self):
        return (self.len - self.seq_len) // self.seq_len


if __name__ == '__main__':
    from tqdm import tqdm
    import numpy as np

    ds = SateliteDataset('./data/train.csv',
                         feature_columns=['x', 'y', 'z'],
                         target_columns=['x_sim', 'y_sim', 'z_sim'])
    for i, d in enumerate(tqdm(ds)):
        a = np.max(d[0])
