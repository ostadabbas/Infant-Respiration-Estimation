import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import math

""" Power Spectral Density MSE loss.
Calculates normalized PSD for output and target. MSE on normalized PSDs.
"""

"""NormPSD code gently borrowed from https://github.com/ToyotaResearchInstitute/RemotePPG 
"""
class NormPSD(nn.Module):
    def __init__(self, fs, high_pass, low_pass):
        super().__init__()
        self.fs = fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x)

        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = torch.add(x[:, :, 0] ** 2, x[:, :, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.fs / 2
        freqs = torch.linspace(0, Fn, x.shape[1])
        use_freqs = torch.logical_and(freqs <= self.high_pass, freqs >= self.low_pass)
        x = x[:,use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)

        return x

class PSD_MSE(nn.Module):
    def __init__(self, fs, high_pass, low_pass):
        super().__init__()
        self.fs = fs
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.psd = NormPSD(self.fs, self.high_pass, self.low_pass)
        self.mse = nn.MSELoss()
        print("Using PSD MSE loss for training.")

    def forward(self, preds, labels):
        pred_psd_norm = self.psd(preds)
        label_psd_norm = self.psd(labels)

        return self.mse(pred_psd_norm, label_psd_norm)
