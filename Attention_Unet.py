import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionGate(nn.Module):
    """
    Additive Attention Gate (AG) from Attention U-Net.
    - x: skip connection features (from encoder)
    - g: gating features (from decoder, coarser scale)
    Produces an attention map to suppress irrelevant skip features.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: channels of gating signal g (decoder feature)
        F_l: channels of skip connection x (encoder feature)
        F_int: intermediate channels (usually F_l // 2)
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        """
        x: skip (B, F_l, H, W)
        g: gate (B, F_g, H', W')  -> will be upsampled to x size if needed
        """
        # If spatial sizes differ, upsample g to x
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)            # (B,1,H,W)
        return x * psi                  # element-wise gating


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        # -------- Encoder --------
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = DoubleConv(in_channels, 64)
        self.down_conv_2 = DoubleConv(64, 128)
        self.down_conv_3 = DoubleConv(128, 256)
        self.down_conv_4 = DoubleConv(256, 512)
        self.down_conv_5 = DoubleConv(512, 1024)

        # -------- Decoder (upsampling) --------
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1  = DoubleConv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2  = DoubleConv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3  = DoubleConv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4  = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

        # -------- Attention Gates on skips --------
        # Gate inputs: g from decoder, x from encoder
        # AG1 gates skip x7 (512ch) using g from up_trans_1 output (512ch)
        self.ag1 = AttentionGate(F_g=512, F_l=512, F_int=512 // 2)
        # AG2 gates skip x5 (256ch) using decoder feat (256ch)
        self.ag2 = AttentionGate(F_g=256, F_l=256, F_int=256 // 2)
        # AG3 gates skip x3 (128ch) using decoder feat (128ch)
        self.ag3 = AttentionGate(F_g=128, F_l=128, F_int=128 // 2)
        # AG4 gates skip x1 (64ch) using decoder feat (64ch)
        self.ag4 = AttentionGate(F_g=64, F_l=64, F_int=64 // 2)

    def forward(self, image):
        # -------- Encoder --------
        x1 = self.down_conv_1(image)      # 64
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)         # 128
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)         # 256
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)         # 512
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)         # 1024 bottleneck

        # -------- Decoder + Attention --------
        d1 = self.up_trans_1(x9)          # 512
        # gate skip x7 with d1
        x7_att = self.ag1(x7, d1)
        d1 = self.up_conv_1(torch.cat([d1, x7_att], dim=1))   # -> 512

        d2 = self.up_trans_2(d1)          # 256
        x5_att = self.ag2(x5, d2)
        d2 = self.up_conv_2(torch.cat([d2, x5_att], dim=1))   # -> 256

        d3 = self.up_trans_3(d2)          # 128
        x3_att = self.ag3(x3, d3)
        d3 = self.up_conv_3(torch.cat([d3, x3_att], dim=1))   # -> 128

        d4 = self.up_trans_4(d3)          # 64
        x1_att = self.ag4(x1, d4)
        d4 = self.up_conv_4(torch.cat([d4, x1_att], dim=1))   # -> 64

        out = self.out(d4)
        return out


if __name__ == '__main__':
    # Example: RGB input, 4 classes
    image = torch.rand((1, 3, 512, 512))
    model = UNet(in_channels=3, num_classes=4)
    output = model(image)
    print("Final shape:", output.shape)
