import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader

from tqdm import tqdm

from models import LSTMPredictor
from dataset import SateliteDataset
from data_transforms import *
from loss import smape_loss
from config import *


ds = SateliteDataset('./data/train.csv',
                     seq_len=SEQ_LEN,
                     feature_columns=FEATURE_COLUMNS,
                     target_columns=TARGET_COLUMNS)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
model = LSTMPredictor(len(FEATURE_COLUMNS), LSTM_UNITS, BATCH_SIZE, tagset_size=len(TARGET_COLUMNS))
loss_function = smape_loss
optimizer = optim.SGD(model.parameters(), lr=1e-3)

prepare_features = transforms.Compose([
    # ToTensor(),
    Preprocess(SEQ_LEN, len(FEATURE_COLUMNS))])

prepare_targets = transforms.Compose([
    # ToTensor()
])
device = torch.device('cuda')

if torch.cuda.is_available():
    model.to(device)

pbar = tqdm(loader)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for step, (features, targets) in enumerate(pbar):
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
