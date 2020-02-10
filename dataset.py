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
        df = pd.read_csv(csv_path)
        self.all_feats = df[feature_columns + ['sat_id']].values
        self.all_targets = df[target_columns + ['sat_id']].values
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.scaler_x = MinMaxScaler().fit(self.all_feats[:, :-1])
        self.scaler_y = MinMaxScaler().fit(self.all_targets[:, :-1])
        self.all_feats[:, :-1] = self.scaler_x.transform(self.all_feats[:, :-1])
        self.all_targets[:, :-1] = self.scaler_x.transform(self.all_targets[:, :-1])
        self.ix = []
        for sat_id, d in df.groupby('sat_id'):
            self.ix += [(sat_id, i - seq_len, i) for i in range(seq_len, d.shape[0])]

    def __getitem__(self, item):
        sat, start, finish = self.ix[item]
        feats = self.all_feats[self.all_feats[:, -1] == sat][start:finish, :-1].astype(np.float32)
        targets = self.all_targets[self.all_targets[:, -1] == sat][finish-1, :-1].astype(np.float32).reshape(1, -1)
        # return self.scaler_x.transform(feats), self.scaler_y.transform(targets)
        return feats, targets

    def __len__(self):
        return len(self.ix)


class InferenceDataset(Dataset):

    def __init__(self, train_csv: str, valid_csv: str, feature_columns: list, target_columns: list, seq_len: int = 5,
                 is_submission: bool = False):
        self.train_df = pd.read_csv(train_csv)
        self.valid_df = pd.read_csv(valid_csv)
        self.ix = []
        for sat, df in self.valid_df.groupby('sat_id'):
            self.ix += [(sat, i) for i in range(len(df))]
        self.feature_columns = feature_columns
        self.target_columns = target_columns if not is_submission else None
        self.is_submission = is_submission
        self.seq_len = seq_len

    def __getitem__(self, item):
        sat, row = self.ix[item]
        if row < self.seq_len-1:
            extra_feats = self.train_df[self.train_df.sat_id == sat].loc[:, self.feature_columns].values[-(self.seq_len - row - 1):]
            features = self.valid_df[self.valid_df.sat_id == sat].loc[:, self.feature_columns].values[:row+1]
            features = np.vstack([extra_feats, features])
        else:
            features = self.valid_df[self.valid_df.sat_id == sat].loc[:, self.feature_columns].values[row-self.seq_len+1:row+1]
        if not self.is_submission:
            targets = self.valid_df[self.valid_df.sat_id == sat].loc[:, self.target_columns].values[row].reshape(1, -1)
            assert features.shape[0] == self.seq_len
            return features.astype(np.float32), targets.astype(np.float32)
        else:
            return features.astype(np.float32)

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
