import torch.nn as nn
import torch.nn.functional as F




class LSTMPredictor(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, batch_size, tagset_size=6):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.sim_prediction = nn.Linear(hidden_dim * batch_size, tagset_size)

    def forward(self, features):

        seq, (hidden_state, cell_state) = self.lstm(features)
        predictions = self.sim_prediction(hidden_state.flatten())
        return predictions
