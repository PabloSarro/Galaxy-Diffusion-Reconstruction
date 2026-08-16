import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Standard Self-Attention block to help the model learn global spatial dependencies.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm(x).view(B, C, H*W).transpose(1,2)         # GroupNorm, reshape as: (B, C, H, W) -> (B, H*W, C).
        att_output, _ = self.mha(h, h, h)                       # Self-attention.
        att_output = att_output.transpose(1,2).view(B, C, H, W) # Reshape back and add residual skip connection.

        return x + att_output


class ResnetBlock(nn.Module):
    """
    A modern Diffusion ResNet block:
    GroupNorm -> SiLU -> Conv -> Dropout -> GroupNorm -> SiLU -> Conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.dropout = nn.Dropout2d(0.10)

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels 
            else nn.Identity()
        )

    def forward(self, x):
        h = self.block1(x)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.shortcut(x)


# DECODER
class Decoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_channels=64):
        super().__init__()

        # =============== U-NET ENCODER ===============
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        # (Downsampling)
        self.down1 = ResnetBlock(base_channels, base_channels)
        self.down2 = ResnetBlock(base_channels, 2*base_channels)
        self.down3 = ResnetBlock(2*base_channels, 4*base_channels)

        # ============ ATTENTION (LOWER LAYERS) ============
        self.att_down = SelfAttention(4*base_channels)
        self.pool = nn.MaxPool2d(2)

        # =============== BOTTLENECK ===============
        self.mid1 = ResnetBlock(4*base_channels, 4*base_channels)
        self.mid_att = SelfAttention(4*base_channels)
        self.mid2 = ResnetBlock(4*base_channels, 4*base_channels)

        # ================= DECODER =================
        # (Upsampling)
        self.up1 = nn.ConvTranspose2d(4*base_channels, 2*base_channels, kernel_size=2, stride=2)
        self.att_up = SelfAttention(4*base_channels)
        self.up_res1 = ResnetBlock(4*base_channels, 2*base_channels) # *4 due to skip connection concatenation

        self.up2 = nn.ConvTranspose2d(2*base_channels, base_channels, kernel_size=2, stride=2)
        self.up_res2 = ResnetBlock(2*base_channels, base_channels)

        # Final projection to match the original shape dimensions (B, C, H, W)
        self.final_res = ResnetBlock(2*base_channels, base_channels)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Initial Convolution
        x = self.init_conv(x)        # (B, 2, 100, 100) --> (B, 64, 100, 100)

        # Downsampling Path
        d1 = self.down1(x)           # Skip 1
        x_pool1 = self.pool(d1)

        d2 = self.down2(x_pool1)     # Skip 2
        x_pool2 = self.pool(d2)

        d3 = self.down3(x_pool2)     # Skip 3
        d3 = self.att_down(d3)

        # Bottleneck
        b = self.mid1(d3)
        b = self.mid_att(b)
        b = self.mid2(b)

        # Upsampling Path
        up1 = self.up1(b)
        up1 = torch.cat([up1, d2], dim=1)   # Concatenate Skip 2
        up1 = self.att_up(up1)
        up1 = self.up_res1(up1)

        up2 = self.up2(up1)
        up2 = torch.cat([up2, d1], dim=1)   # Concatenate Skip 1
        up2 = self.up_res2(up2)

        # Final Convolution
        out = torch.cat([up2, x], dim=1)
        out = self.final_res(out)
        return self.final_conv(out)


def decoder_loss(pred_x0, true_x0, method="MSE", alpha=0):
    """
    Computes the reconstruction loss between the predicted and true clean images.

    Methods:
    - MSE
    - MSE_MAE
    - MSE_Grad
    - Charbonnier
    - Weighted
    """

    if method=="MSE":
        return F.mse_loss(pred_x0, true_x0)
    
    elif method=="MSE_MAE":
        return 0.5*F.mse_loss(pred_x0, true_x0) + 0.5*F.l1_loss(pred_x0, true_x0) # maybe try different ratios.

    elif method=="MSE_Grad":
        dx_pred = pred_x0[:, :, :, 1:] - pred_x0[:, :, :, :-1]
        dx_true = true_x0[:, :, :, 1:] - true_x0[:, :, :, :-1]

        dy_pred = pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :]
        dy_true = true_x0[:, :, 1:, :] - true_x0[:, :, :-1, :]

        grad_loss = F.l1_loss(dx_pred, dx_true) + F.l1_loss(dy_pred, dy_true)
        mse_loss = F.mse_loss(pred_x0, true_x0)

        return mse_loss + 0.1*grad_loss

    elif method=="Charbonnier":
        eps = 1e-3
        diff = pred_x0 - true_x0
        return torch.mean(torch.sqrt(diff*diff + eps**2))

    elif method=="Weighted":
        weight = 1 + alpha*torch.abs(true_x0)
        return torch.mean(weight * (pred_x0-true_x0)**2)

    else:
        raise ValueError(f"Unknown loss method: {method}")