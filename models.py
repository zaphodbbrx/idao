import torch
import torch.nn as nn
import numpy as np


class InferenceRunner:

    def __init__(self, model, prediction_mode='simple', scaler_x=None, scaler_y=None):
        self.model = model
        self.prediction_mode = prediction_mode
        self.last_prediction = None
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    def predict_rec(self, features):
        pass

    def predict_simple(self, features):
        features = np.stack([self.scaler_x.transform(features.cpu()[i]) for i in range(features.shape[0])], 0).astype(np.float32)
        features = torch.from_numpy(features).to('cuda' if torch.cuda.is_available() else 'cpu')
        preds = self.model(features)
        if self.scaler_y:
            preds = self.scaler_y.inverse_transform(preds.detach().cpu().numpy())
        return preds

    def __call__(self, features):
        if self.prediction_mode == 'recurrent':
            return self.predict_rec(features)
        elif self.prediction_mode == 'simple':
            return self.predict_simple(features)
        else:
            raise ValueError('Invalid prediction mode. Expected recurrent or simple')


class LSTMPredictor(nn.Module):

    def __init__(self, num_features, hidden_dim, tagset_size=6):
        super(LSTMPredictor, self).__init__()
        self.num_layers = 1
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.sim_prediction = nn.Linear(hidden_dim, tagset_size)

    def forward(self, features):
        h0 = torch.zeros(1, features.size(0), self.hidden_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
        c0 = torch.zeros(1, features.size(0), self.hidden_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
        seq, (hidden_state, cell_state) = self.lstm(features, (h0, c0))
        hidden_state = hidden_state.detach()
        predictions = self.sim_prediction(hidden_state.squeeze())
        predictions = predictions + features[:, -1].squeeze()
        return predictions


class Conv1dPredictor(nn.Module):

    def __init__(self, num_features, target_size=6):
        super(Conv1dPredictor, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 256, 2, stride=1)
        self.conv2 = nn.Conv1d(256, 256, 2, stride=1)
        self.conv3 = nn.Conv1d(256, 256, 2, stride=1)
        self.conv4 = nn.Conv1d(256, target_size, 2, stride=1)
        self.preds = nn.Linear(21 * target_size, target_size)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(target_size)

        self.pool1 = nn.MaxPool1d(1)
        self.pool2 = nn.MaxPool1d(1)
        self.pool3 = nn.MaxPool1d(1)
        self.pool4 = nn.MaxPool1d(1)

    def forward(self, features):
        x = self.conv1(features.permute(0, 2, 1))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool4(x)
        predictions = self.preds(x.view(x.size(0), -1))
        return predictions


class MLPPredictor(nn.Module):

    def __init__(self, num_features, seq_len, hidden_dim, target_size=6):
        super(MLPPredictor, self).__init__()
        self.num_layers = 1
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.seq_len = seq_len
        self.dense1 = nn.Linear(seq_len * num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dense3 = nn.Linear(hidden_dim // 2, target_size)
        self.target_size = target_size
        self.relu = nn.ReLU()

    def forward(self, features):
        x = features.view(-1, self.seq_len * self.num_features)
        x = self.dense1(x)
        # x = self.bn1(x)
        x = self.dense2(x)
        # x = self.bn2(x)
        predictions = self.dense3(x)
        predictions = predictions.view(-1, self.target_size)
        return predictions
