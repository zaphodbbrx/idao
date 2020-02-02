import time

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
        self.filereader = iter(pd.read_csv(csv_path, chunksize=100000, error_bad_lines=False, sep=','))
        df_t = pd.read_csv(csv_path, nrows=100000)
        self.scaler_x = MinMaxScaler().fit(df_t[feature_columns].values)
        self.scaler_y = MinMaxScaler().fit(df_t[target_columns].values)
        self.df = None
        self.datagen = None
        self.df_idx = None
        self.reset_iter()

    def reset_iter(self):
        self.df = next(self.filereader)
        self.buffer_x = self.scaler_x.transform(self.df[self.feature_columns].values)
        self.buffer_y = self.scaler_y.transform(self.df[self.target_columns].values)
        self.df_idx = 0

    def __getitem__(self, item):
        if self.df_idx < self.df.shape[0] - self.seq_len -1:
            self.df_idx += 1
        else:
            self.reset_iter()
        features = self.buffer_x[self.df_idx:self.df_idx + self.seq_len].astype(np.float32)
        targets = self.buffer_y[self.df_idx + self.seq_len].reshape(1, -1).astype(np.float32)
        return features, targets

    def __len__(self):
        return (self.len - self.seq_len) // self.seq_len


class InferenceDataset(Dataset):

    def __init__(self, train_csv: str, valid_csv: str, feature_columns: list, target_columns: list, seq_len: int = 5):
        self.train_df = pd.read_csv(train_csv)
        self.valid_df = pd.read_csv(valid_csv)
        self.ix = []
        for sat, df in self.valid_df.groupby('sat_id'):
            self.ix += [(sat, i) for i in range(len(df))]
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.seq_len = seq_len

    def __getitem__(self, item):
        sat, row = self.ix[item]
        if row < self.seq_len:
            extra_feats = self.train_df[self.train_df.sat_id == sat].loc[:, self.feature_columns].values[-(self.seq_len - row):]
            features = self.valid_df[self.valid_df.sat_id == sat].loc[:, self.feature_columns].values[:row]
            features = np.vstack([extra_feats, features])
        else:
            features = self.valid_df[self.valid_df.sat_id == sat].loc[:, self.feature_columns].values[row-self.seq_len:row]
        targets = self.valid_df[self.valid_df.sat_id == sat].loc[:, self.target_columns].values[row].reshape(1, -1)
        assert features.shape[0] == self.seq_len
        return features.astype(np.float32), targets.astype(np.float32)

    def __len__(self):
        return len(self.ix)


if __name__ == '__main__':
    from tqdm import tqdm
    import numpy as np

    ds = SateliteDataset('./data/split_train.csv',
                          feature_columns=['x', 'y', 'z'],
                          target_columns=['x_sim', 'y_sim', 'z_sim'])
    start = time.time()
    a = ds[0]
    end = time.time()
    print(end-start)
