""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from functorch import vmap
import numpy as np


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode="circular"),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode="circular"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], mode='circular')
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) #add high pass filter exp(-k_h/k)FFT(k), f(k=0)=0, k = |k|

    def forward(self, x, k_h = 3.0):
        output = torch.vmap(torch.vmap(lambda img: self.high_pass(img, k_h), in_dims=0), in_dims=0)(self.conv(x))
        '''
        output = self.conv(x)
        output = output - output.mean(dim=(2, 3), keepdim=True)
        print('no high pass')
        '''
        return output

    def high_pass(self, image, k_h):
        H, W = image.shape
        L = 25. #Mpc/c
        
        kx = torch.fft.fftfreq(H, d=1. / (H * 2 * np.pi / L)).to(image.device)
        ky = torch.fft.fftfreq(W, d=1. / (W * 2 * np.pi / L)).to(image.device)
        kx, ky = torch.meshgrid(kx, ky, indexing="ij")
        k = torch.sqrt(kx**2 + ky**2)
    
        # Apply the high-pass filter
        filter_mask = torch.exp(-k_h / (k + 1e-8))  # Add small constant to avoid division by zero 1-e^(-k^2/k_h^2)
        filter_mask[0, 0] = 0.0  # Set zero mode to zero
    
        # Perform Fourier transform, apply filter, and inverse transform
        image_fft = fft.fft2(image)
        image_fft_filtered = image_fft * filter_mask
        filtered_image = fft.ifft2(image_fft_filtered).real
    
        return filtered_image


class FourierFeatures(nn.Module):
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        assert len(x.shape) >= 2

        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0**freqs_exponent * 2 * pi  # (F, )
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, X1, X2, ...)
        features = features.flatten(1, 2)  # (B, F * C, X1, X2, ...)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)