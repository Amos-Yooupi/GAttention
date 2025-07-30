import torch


def fourier_loss(true, predict):
    """compare in freq domain"""
    true_fft = torch.fft.fft(true, dim=0)
    predict_fft = torch.fft.fft(predict, dim=0)
    loss = torch.mean(torch.abs(true_fft - predict_fft))
    return loss.item()
