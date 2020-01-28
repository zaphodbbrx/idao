import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils

from tqdm import tqdm

from models import LSTMPredictor
from dataset import SateliteDataset
from data_transforms import *
from config import *


ds = SateliteDataset('./data/train.csv', SEQ_LEN)
model = LSTMPredictor(N_FEATURES, LSTM_UNITS)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-6)
prepare_features = transforms.Compose([ToTensor(), Preprocess(SEQ_LEN, N_FEATURES)])
prepare_targets = transforms.Compose([ToTensor()])
device = torch.device('cuda')

if torch.cuda.is_available():
    model.to(device)

pbar = tqdm(ds)

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
        if step % 100 == 0:
            pbar.set_description(f'MSE: {loss}')
