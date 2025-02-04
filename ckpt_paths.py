# Unconditional
path_UNet_10_epochs = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/7324872b734c47469bd8054d8263538d/checkpoints/best_model-epoch=9-step=10000.ckpt'


# FiLMed condition
## CMD 2D
path_UNet_Film_10_epochs = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/44ed9965667345bda39aaaad1cd9d543/checkpoints/latest-epoch=9-step=8750.ckpt'
path_UNet_Film_10_epochs_fourier_loss = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/106e0975f82c4de1b3f2f7e1f80527ad/checkpoints/latest-epoch=9-step=8750.ckpt'
path_UNet_Film_50_epochs_fourier_loss_2_10 = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/9a96658e7271445ebc42cbe7cf86fa99/checkpoints/latest-epoch=49-step=43700.ckpt'
## 25 thickness maps
path_UNet_Film_10_epochs_ONLY_l1_loss_25_thick = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/62ce892d8ce84414ad1d864f0b3c09a3/checkpoints/latest-epoch=9-step=1750.ckpt'

###Fourier loss
path_UNet_Film_10_epochs_ONLY_fourier_loss_3_10_zeros_elsewhere_25_thick = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/5ef90d60aaca4012837cd4275273e171/checkpoints/latest-epoch=9-step=1750.ckpt'
path_UNet_Film_10_epochs_5e3_ONLY_fourier_loss_2_20_zeros_elsewhere_25_thick = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/8a4577d4dcd14186883e9503a5018238/checkpoints/latest-epoch=9-step=1750.ckpt'
path_UNet_Film_10_epochs_1e3_ONLY_fourier_loss_3_10_25_thick = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/1ef6303d65494dd7b8cc77bdad1650c6/checkpoints/latest-epoch=9-step=1750.ckpt'
path_UNet_Film_10_epochs_1e4_ONLY_fourier_loss_2_20_zeros_elsewhere_25_thick = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/dc37c367be9b4f29b488048917aa5984/checkpoints/latest-epoch=9-step=1750.ckpt'
path_UNet_Film_10_epochs_Fourier_ONLY_weight_1e7_3_10_high_pass_lr_1e3_adamw_1e2_25_thickness = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/92697b2dc3e1460389ff1ccea6f93b28/checkpoints/latest-epoch=9-step=1750.ckpt'
path_UNet_Film_10_epochs_Fourier_ONLY_weight_1e7_3_10_Butterworth_high_pass_lr_1e3_adamw_1e2_25_thickness = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/82e98acaf3924668946adf6e4416961f/checkpoints/latest-epoch=9-step=1750.ckpt'
Unet_Film_high_pass_Fourier_ONLY_1e5_3_10_Butterworth_lr_1e3_adamw_1e2_25_thickness_Nbody_overdensity = '/pscratch/sd/l/lindajin/LOGS/Nbody_Hydro/72980c14e46a423994686ce1399048f8/checkpoints/latest-epoch=9-step=1750.ckpt'


###L1+Fourier
path_UNet_Film_10_epochs_Fourier_L1__2_20_high_pass_25_thickness = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/96071eeeabd64a0da32d6e71b78e41a9/checkpoints/latest-epoch=9-step=1750.ckpt'
path_UNet_Film_10_epochs_L1_epoch2_Fourier_weight_5e5_3_10_Butterworth_high_pass_lr_1e3_adamw_1e2_25_thickness = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/68d21ff455be4c49bc266767ff0f9550/checkpoints/latest-epoch=9-step=1750.ckpt'
path_UNet_Film_10_epochs_L1_epoch5_Fourier_weight_1e5_3_10_Butterworth_high_pass_lr_1e3_adamw_1e2_25_thickness = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/d897ce699f2242889e3f18293dea323d/checkpoints/latest-epoch=9-step=1750.ckpt'
path_UNet_Film_10_epochs_L1_epoch10_Fourier_weight_1e5_1_10_Butterworth_high_pass_lr_1e3_adamw_1e2_25_thickness='/pscratch/sd/l/lindajin/LOGS/baryonize_DM/dca6e98452524e9c9797f56d152fe01c/checkpoints/latest-epoch=14-step=2600.ckpt'
path_UNet_Film_10_epochs_L1_epoch2_Fourier_weight_1e5_3_10_Butterworth_lr_1e3_adamw_1e2_25_thickness = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/063589d4efa3418d9728cd32a7ab2ee3/checkpoints/latest-epoch=9-step=1750.ckpt'
path_UNet_Film_10_epochs_L1_epoch2_Fourier_weight_5e5_3_10_Butterworth_high_pass_lr_1e3_adamw_1e2_25_thickness='/pscratch/sd/l/lindajin/LOGS/Nbody_Hydro/b7e663e255bf4e649cee1e66666e2f12/checkpoints/latest-epoch=9-step=1750.ckpt'


#BEST
path_UNet_Film_10_epochs_L1_epoch2_Fourier_weight_1e5_3_10_Butterworth_high_pass_lr_1e3_adamw_1e2_25_thickness_Nbody_overdensity = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/14353a9080854cd69f7119e0d2ec4718/checkpoints/latest-epoch=9-step=1750.ckpt'


# UNet + Divij's TF
path_UNet_50_epochs = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/f82bd19f11ba472c827e20906e0f0fd5/checkpoints/latest-epoch=49-step=10000.ckpt'
path_UNet_10_epochs_circ_padding = '/pscratch/sd/l/lindajin/linda_hydro_emulator/baryonize_DM/7de78940c94547ad878fde7565871c98/checkpoints/latest-epoch=9-step=2000.ckpt'
path_UNetVDM_10_epochs = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/2636f1e92950444a94aff66298537323/checkpoints/latest-epoch=9-step=2000.ckpt'
path_UNet_high_pass ='/pscratch/sd/l/lindajin/LOGS/baryonize_DM/7f09ef7fd64d4eac8c2fdf6d960a0cc1/checkpoints/latest-epoch=9-step=2000.ckpt'
path_UNet_Film_10_epochs_ONLY_fourier_loss_3_10_zeros_elsewhere = '/pscratch/sd/l/lindajin/LOGS/baryonize_DM/5dac20b8da8145bfbbf0feee77f69ffb/checkpoints/latest-epoch=9-step=2000.ckpt'