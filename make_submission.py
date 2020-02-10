import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from tqdm import tqdm

from models import *
from dataset import SateliteDataset, InferenceDataset
from data_transforms import *
from loss import smape_loss
from utils import smape
from config import *

model_path = '/home/lsm/projects/kaggle/idao/model.pth'


ds_train = SateliteDataset('./data/split_train.csv',
                           seq_len=SEQ_LEN,
                           feature_columns=FEATURE_COLUMNS,
                           target_columns=TARGET_COLUMNS)
ds_valid = InferenceDataset('./data/train.csv', './data/Track 1/test.csv',
                            seq_len=SEQ_LEN,
                            feature_columns=FEATURE_COLUMNS,
                            target_columns=TARGET_COLUMNS,
                            is_submission=True
                            )

loader_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=False)

pbar = tqdm(loader_valid)
# model = LSTMPredictor(len(FEATURE_COLUMNS), LSTM_UNITS, tagset_size=len(TARGET_COLUMNS))
model = Conv1dPredictor(len(FEATURE_COLUMNS), target_size=len(TARGET_COLUMNS))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.cuda()
model.eval()
runner = InferenceRunner(model, scaler_x=ds_train.scaler_x, scaler_y=ds_train.scaler_y)
df_sub = pd.read_csv('./data/Track 1/submission.csv')
prepare_features = transforms.Compose([
    # ToTensor(),
    # Preprocess(SEQ_LEN, len(FEATURE_COLUMNS))
    ])

prepare_targets = transforms.Compose([
    # ToTensor()
])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
all_preds = []
for features in pbar:
    preprocessed_features = prepare_features(features)
    if torch.cuda.is_available():
        preprocessed_features = preprocessed_features.to(device)
    predictions = runner(preprocessed_features)
    all_preds.append(predictions)

all_preds = np.vstack(all_preds)

df_sub.iloc[:len(all_preds), 1:] = all_preds

df_sub.to_csv('data/sub.csv', index=False)