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
        return sample.permute(1, 0, 2).float()

