import torch
from torch.utils.data import Dataset
import pandas as pd


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


class SateliteDataset(Dataset):

    def __init__(self, csv_path, seq_len=5):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.len = file_len(csv_path)
        self.colnames = pd.read_csv(csv_path, nrows=1).columns.tolist()

    def __getitem__(self, item):
        df = pd.read_csv(self.csv_path, skiprows=item+1, nrows=self.seq_len, names=self.colnames)
        features = df[['x', 'y', 'z', 'Vx', 'Vy', 'Vz']].values
        targets = df[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']].values[-1]
        return features, targets

    def __len__(self):
        return self.len


if __name__ == '__main__':
    ds = SateliteDataset('./data/train.csv')
    a = ds[0]
    pass