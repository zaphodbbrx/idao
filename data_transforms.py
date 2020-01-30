import torch


__all__ = ['ToTensor', 'Preprocess']


class ToTensor:

    def __call__(self, sample):
        return torch.from_numpy(sample).float()


class Preprocess:

    def __init__(self, seq_len, n_features):
        self.n_features = n_features
        self.seq_len = seq_len

    def __call__(self, sample):
        return sample.view(self.seq_len, -1, self.n_features).float()

