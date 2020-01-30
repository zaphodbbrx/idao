import torch


def smape_loss(true, predicted, device='cuda'):
    epsilon = 0.1
    true_o = true
    pred_o = predicted
    summ = torch.max(torch.abs(true_o) + torch.abs(pred_o) + epsilon, torch.tensor(0.5 + epsilon).to(device))
    smape = torch.abs(pred_o - true_o) / summ
    return torch.mean(smape)
