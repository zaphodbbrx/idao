import os
import pandas as pd
import numpy as np


def split(csv_path):
    all_data = pd.read_csv(csv_path)
    train, val = [], []
    for sat_id, df in all_data.groupby('sat_id'):
        if df.shape[0] <= 200:
            continue
        train.append(df.iloc[:-50])
        val.append(df.iloc[-50:])
    df_train = pd.concat(train)
    df_val = pd.concat(val)
    df_train.to_csv(os.path.join(os.path.dirname(csv_path), 'split_train.csv'))
    df_val.to_csv(os.path.join(os.path.dirname(csv_path), 'split_val.csv'))


def smape(forecasts, actuals):
    return 100 / forecasts.shape[0] * 2 * np.sum(np.abs(forecasts - actuals) / (np.abs(forecasts) + np.abs(actuals)))


if __name__ == '__main__':
    split(csv_path='./data/train.csv')
    pass
