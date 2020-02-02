import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from models import LSTMPredictor, InferenceRunner
from dataset import SateliteDataset, InferenceDataset
from data_transforms import *
from loss import smape_loss
from utils import smape
from config import *


ds_train = SateliteDataset('./data/split_train.csv',
                           seq_len=SEQ_LEN,
                           feature_columns=FEATURE_COLUMNS,
                           target_columns=TARGET_COLUMNS)
ds_valid = InferenceDataset('./data/split_train.csv', './data/split_val.csv',
                            seq_len=SEQ_LEN,
                            feature_columns=FEATURE_COLUMNS,
                            target_columns=TARGET_COLUMNS)
loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
loader_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=False)
model = LSTMPredictor(len(FEATURE_COLUMNS), LSTM_UNITS, BATCH_SIZE, tagset_size=len(TARGET_COLUMNS))

loss_function = smape_loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
writer = SummaryWriter(log_dir='./train_logs')


prepare_features = transforms.Compose([
    # ToTensor(),
    Preprocess(SEQ_LEN, len(FEATURE_COLUMNS))])

prepare_targets = transforms.Compose([
    # ToTensor()
])
device = torch.device('cuda')

if torch.cuda.is_available():
    model.to(device)

step = 0
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    pbar = tqdm(loader_train)
    model.train()
    for features, targets in pbar:
        step += 1
        model.zero_grad()
        preprocessed_features = prepare_features(features)
        preprocessed_targets = prepare_targets(targets)

        if torch.cuda.is_available():
            preprocessed_features = preprocessed_features.to(device)
            preprocessed_targets = preprocessed_targets.to(device)

        predictions = model(preprocessed_features)

        loss = loss_function(predictions, preprocessed_targets)
        loss.backward()
        optimizer.step()
        pbar.set_description(f'loss: {loss}')
        if step // (100 / BATCH_SIZE):
            writer.add_scalar('Loss/SMAPE', loss, step)
    pbar = tqdm(loader_valid)
    model.eval()
    runner = InferenceRunner(model, scaler_x=ds_train.scaler_x, scaler_y=ds_train.scaler_y)
    smape = 0
    for features, targets in pbar:
        preprocessed_features = prepare_features(features)
        if torch.cuda.is_available():
            preprocessed_features = preprocessed_features.to(device)

        predictions = runner(preprocessed_features)
        smape += np.mean(np.abs(predictions - targets.numpy()) / (np.abs(predictions) + np.abs(targets.numpy()))) / len(loader_valid)
    print(f"Epoch: {epoch} SMAPE: {1 - smape:.5f}")
