""" Full assembly of the parts to form the complete network """
import sys
import torch.nn as nn
from unet_parts import *

class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, params): #params has shape (N_batch, N_ch*2)
    params = params.view(params.size(0), -1, 2) #(N_batch, N_ch, 2)
    gammas = params[:, :, 0] 
    betas = params[:, :, 1]
    gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
    betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    return (gammas * x) + betas

class UNetFiLMNoSkip(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear: bool = False, use_fourier_features: bool = False):
        super().__init__()
        self.use_fourier_features = use_fourier_features
        if use_fourier_features:
            self.fourier_features = FourierFeatures(
                first=6.0,
                last=7.0,
                step=1,
            )
        if use_fourier_features:
            n_channels *= 1 + self.fourier_features.num_features

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.film = FiLM()
        self.MLP = nn.Sequential(
                nn.Linear(6, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, 2048),
                nn.ReLU(),
                nn.Linear(2048, 5888)#*channel each layer
                    )
       # Apply zero-weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

                
    def forward(self, x, cond, inverse_blue_filter=False, high_pass = False, k_h = 3.0, order = 1.0): #cond(6,)-->MLP score(N,)=weight+bias
        param = self.MLP(cond)
        x = self.maybe_concat_fourier(x)    
        x1 = self.inc(x)
        x1 = self.film(x1, param[:,:128])
        x2 = self.down1(x1)
        x2 = self.film(x2, param[:,128:384])
        x3 = self.down2(x2)
        x3 = self.film(x3, param[:,384:896])
        x4 = self.down3(x3)
        x4 = self.film(x4, param[:,896:1920])
        x5 = self.down4(x4)
        x5 = self.film(x5, param[:,1920:3968])
        x = self.up1(x5)#, x4)
        x = self.film(x, param[:,3968:4992])
        x = self.up2(x)#, x3)
        x = self.film(x, param[:,4992:5504])
        x = self.up3(x)#, x2)
        x = self.film(x, param[:,5504:5760])
        x = self.up4(x)#, x1)
        x = self.film(x, param[:,5760:5888])
        logits = self.outc(x, inverse_blue_filter, high_pass, k_h, order)
        return logits


    def maybe_concat_fourier(self, x):
        if self.use_fourier_features:
            return torch.cat([x, self.fourier_features(x)], dim=1)
        return x
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNetFiLM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear: bool = False, use_fourier_features: bool = False):
        super().__init__()
        self.use_fourier_features = use_fourier_features
        if use_fourier_features:
            self.fourier_features = FourierFeatures(
                first=6.0,
                last=7.0,
                step=1,
            )
        if use_fourier_features:
            n_channels *= 1 + self.fourier_features.num_features

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up_SkipConnection(1024, 512 // factor, bilinear))
        self.up2 = (Up_SkipConnection(512, 256 // factor, bilinear))
        self.up3 = (Up_SkipConnection(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.film = FiLM()
        self.MLP = nn.Sequential(
                nn.Linear(6, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, 2048),
                nn.ReLU(),
                nn.Linear(2048, 5888)#*channel each layer
                    )
       # Apply zero-weight initialization
        self._initialize_weights()
        print('no skip connection on last layer')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

                
    def forward(self, x, cond, inverse_blue_filter=False, high_pass = False, k_h = 3.0, order = 1.0): #cond(6,)-->MLP score(N,)=weight+bias
        param = self.MLP(cond)
        x = self.maybe_concat_fourier(x)    
        x1 = self.inc(x)
        x1 = self.film(x1, param[:,:128])
        x2 = self.down1(x1)
        x2 = self.film(x2, param[:,128:384])
        x3 = self.down2(x2)
        x3 = self.film(x3, param[:,384:896])
        x4 = self.down3(x3)
        x4 = self.film(x4, param[:,896:1920])
        x5 = self.down4(x4)
        x5 = self.film(x5, param[:,1920:3968])
        x = self.up1(x5, x4)
        x = self.film(x, param[:,3968:4992])
        x = self.up2(x, x3)
        x = self.film(x, param[:,4992:5504])
        x = self.up3(x, x2)
        x = self.film(x, param[:,5504:5760])
        x = self.up4(x) #, x1)
        x = self.film(x, param[:,5760:5888])
        logits = self.outc(x, inverse_blue_filter, high_pass, k_h, order)
        return logits


    def maybe_concat_fourier(self, x):
        if self.use_fourier_features:
            return torch.cat([x, self.fourier_features(x)], dim=1)
        return x
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


        
class ResBlockFiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.double_conv = DoubleConv(channels, channels)
        self.film = FiLM()
        
    def forward(self, x, film_params):
        identity = x
        out = self.double_conv(x)
        out = self.film(out, film_params)
        return F.relu(out + identity)

class ResNetFiLM(nn.Module):
    def __init__(self, n_channels, n_classes, use_fourier_features: bool = False, high_pass: bool = False):
        super().__init__()
        self.use_fourier_features = use_fourier_features
        if use_fourier_features:
            self.fourier_features = FourierFeatures(
                first=-2.0,
                last=1,
                step=1,
            )
            n_channels *= 1 + self.fourier_features.num_features
            
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Initial feature extraction (n_channels -> 64)
        self.inc = DoubleConv(n_channels, 64)
        
        # Downsampling to capture context (64 -> 128 -> 256)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        
        # ResNet blocks for small-scale feature processing (256 channels)
        self.res_blocks = nn.ModuleList([
            ResBlockFiLM(256) for _ in range(8)
        ])
        
        # Final projection (256 -> n_classes)
        self.outc = OutConv(256, n_classes*16, high_pass)

        self.upsample =  nn.Sequential(
            nn.PixelShuffle(upscale_factor=4),  # Upscale x4
            nn.ReLU(inplace=True)
        )
        
        # Calculate FiLM parameters size:
        # 64 channels (inc) -> 128 params
        # 128 channels (down1) -> 256 params
        # 256 channels (down2) -> 512 params
        # 8 ResBlocks with 256 channels each -> 8 * 512 params
        total_film_params = 128 + 256 + 512 + (8 * 512)
        
        # FiLM conditioning
        self.film = FiLM()
        self.MLP = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, total_film_params)
        )
        
    def forward(self, x, cond=None):
        param = self.MLP(cond)
        param_idx = 0
        
        # Apply Fourier features if enabled
        x = self.maybe_concat_fourier(x)
        
        # Initial feature extraction
        x = self.inc(x)
        x = self.film(x, param[:, param_idx:param_idx+128])
        param_idx += 128
        
        # First downsampling
        x = self.down1(x)
        x = self.film(x, param[:, param_idx:param_idx+256])
        param_idx += 256
        
        # Second downsampling
        x = self.down2(x)
        x = self.film(x, param[:, param_idx:param_idx+512])
        param_idx += 512
        
        # ResNet blocks with FiLM
        for res_block in self.res_blocks:
            x = res_block(x, param[:, param_idx:param_idx+512])
            param_idx += 512
        
        # High-pass filtered output
        x = self.outc(x)

        # Upsample back to original resolution
        x = self.upsample(x)
        
        return x
    
    def maybe_concat_fourier(self, x):
        if self.use_fourier_features:
            return torch.cat([x, self.fourier_features(x)], dim=1)
        return x
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        for i in range(len(self.res_blocks)):
            self.res_blocks[i] = torch.utils.checkpoint.checkpoint(self.res_blocks[i])
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)