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
        features = np.stack([self.scaler_x.transform(features.cpu()[:, i]) for i in range(features.shape[1])], 1).astype(np.float32)
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

    def __init__(self, embedding_dim, hidden_dim, batch_size, tagset_size=6):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.sim_prediction = nn.Linear(hidden_dim, tagset_size)

    def forward(self, features):
        seq, (hidden_state, cell_state) = self.lstm(features)
        predictions = self.sim_prediction(cell_state.squeeze())
        predictions = predictions + features[-1].squeeze()
        return predictions
