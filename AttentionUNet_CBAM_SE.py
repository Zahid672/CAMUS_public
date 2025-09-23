# AttentionUNet_CBAM_SE.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- Attention primitives --------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class ChannelAttention(nn.Module):
    """Channel attention used by CBAM (avg + max pooling)."""
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_w = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_w = self.mlp(F.adaptive_max_pool2d(x, 1))
        w = self.sig(avg_w + max_w)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention used by CBAM (channel-pooled)."""
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        w = self.sig(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """CBAM = ChannelAttention -> SpatialAttention."""
    def __init__(self, ch, r=16, spatial_ks=7):
        super().__init__()
        self.ca = ChannelAttention(ch, r=r)
        self.sa = SpatialAttention(kernel_size=spatial_ks)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# -------------------- Basic conv blocks --------------------
class DoubleConv(nn.Module):
    """Two 3x3 convs + BN + ReLU with optional CBAM/SE at the end."""
    def __init__(self, in_ch, out_ch, use_cbam=False, cbam_r=16, use_se=False, se_r=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_ch, r=cbam_r) if use_cbam else None
        self.se   = SEBlock(out_ch, r=se_r) if use_se   else None

    def forward(self, x):
        x = self.block(x)
        if self.cbam is not None:
            x = self.cbam(x)
        if self.se is not None:
            x = self.se(x)
        return x


class AttentionGate(nn.Module):
    """
    Additive Attention Gate (AG) from Attention U-Net with an optional CBAM
    pre-enhancement on the skip feature x before the additive gating.
    """
    def __init__(self, F_g, F_l, F_int, use_cbam_on_skip=True, cbam_r=16):
        super().__init__()
        self.use_cbam_on_skip = use_cbam_on_skip
        self.cbam_skip = CBAM(F_l, r=cbam_r) if use_cbam_on_skip else None

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
        # Align spatial sizes (upsample g to x if needed)
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.cbam_skip is not None:
            x = self.cbam_skip(x)  # enhance skip before gating

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # (B,1,H,W)
        return x * psi       # gate skip by attention map


# -------------------- Attention U-Net with CBAM/SE --------------------
class UNet(nn.Module):
    """
    Attention U-Net enhanced with CBAM (encoder/decoder), optional SE in bottleneck,
    and CBAM-augmented attention gates on the skips.
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        # Where to enable CBAM / SE:
        enc_cbam_stages=(False, True, True, True, False),   # length for 5 encoder convs
        dec_cbam_stages=(True, True, True, False),          # length for 4 decoder convs
        gate_use_cbam=True,
        bottleneck_se=True,
        cbam_reduction=16,
        se_reduction=16
    ):
        """
        enc_cbam_stages: tuple of 5 booleans for DoubleConv(1..5)
        dec_cbam_stages: tuple of 4 booleans for decoder DoubleConv(1..4)
        gate_use_cbam:   apply CBAM on skip features inside attention gates
        bottleneck_se:   apply SE on the 1024-ch bottleneck
        """
        super().__init__()
        assert len(enc_cbam_stages) == 5, "enc_cbam_stages must have 5 flags"
        assert len(dec_cbam_stages) == 4, "dec_cbam_stages must have 4 flags"

        # -------- Encoder --------
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = DoubleConv(in_channels,  64, use_cbam=enc_cbam_stages[0], cbam_r=cbam_reduction)
        self.down_conv_2 = DoubleConv(64,  128, use_cbam=enc_cbam_stages[1], cbam_r=cbam_reduction)
        self.down_conv_3 = DoubleConv(128, 256, use_cbam=enc_cbam_stages[2], cbam_r=cbam_reduction)
        self.down_conv_4 = DoubleConv(256, 512, use_cbam=enc_cbam_stages[3], cbam_r=cbam_reduction)
        self.down_conv_5 = DoubleConv(512, 1024, use_cbam=enc_cbam_stages[4], cbam_r=cbam_reduction,
                                      use_se=bottleneck_se, se_r=se_reduction)

        # -------- Decoder (upsampling) --------
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1  = DoubleConv(1024, 512, use_cbam=dec_cbam_stages[0], cbam_r=cbam_reduction)

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2  = DoubleConv(512, 256, use_cbam=dec_cbam_stages[1], cbam_r=cbam_reduction)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3  = DoubleConv(256, 128, use_cbam=dec_cbam_stages[2], cbam_r=cbam_reduction)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4  = DoubleConv(128, 64, use_cbam=dec_cbam_stages[3], cbam_r=cbam_reduction)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

        # -------- Attention Gates on skips --------
        self.ag1 = AttentionGate(F_g=512, F_l=512, F_int=512 // 2,
                                 use_cbam_on_skip=gate_use_cbam, cbam_r=cbam_reduction)
        self.ag2 = AttentionGate(F_g=256, F_l=256, F_int=256 // 2,
                                 use_cbam_on_skip=gate_use_cbam, cbam_r=cbam_reduction)
        self.ag3 = AttentionGate(F_g=128, F_l=128, F_int=128 // 2,
                                 use_cbam_on_skip=gate_use_cbam, cbam_r=cbam_reduction)
        self.ag4 = AttentionGate(F_g=64,  F_l=64,  F_int=64  // 2,
                                 use_cbam_on_skip=gate_use_cbam, cbam_r=cbam_reduction)

    def forward(self, image):
        # -------- Encoder --------
        x1 = self.down_conv_1(image)     # 64
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)        # 128
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)        # 256
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)        # 512
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)        # 1024 bottleneck

        # -------- Decoder + Attention (with CBAM-enhanced skips) --------
        d1 = self.up_trans_1(x9)         # -> 512
        x7_att = self.ag1(x7, d1)        # gate skip with CBAM-augmented AG
        d1 = self.up_conv_1(torch.cat([d1, x7_att], dim=1))  # -> 512

        d2 = self.up_trans_2(d1)         # -> 256
        x5_att = self.ag2(x5, d2)
        d2 = self.up_conv_2(torch.cat([d2, x5_att], dim=1))  # -> 256

        d3 = self.up_trans_3(d2)         # -> 128
        x3_att = self.ag3(x3, d3)
        d3 = self.up_conv_3(torch.cat([d3, x3_att], dim=1))  # -> 128

        d4 = self.up_trans_4(d3)         # -> 64
        x1_att = self.ag4(x1, d4)
        d4 = self.up_conv_4(torch.cat([d4, x1_att], dim=1))  # -> 64

        out = self.out(d4)
        return out


# -------------------- quick self-test --------------------
if __name__ == "__main__":
    # Example: 1-channel CAMUS image, 4 classes
    x = torch.randn(2, 1, 256, 256)
    model = UNet(
        in_channels=1,
        num_classes=4,
        # Enable CBAM on deeper encoder stages & decoder,
        # use SE at bottleneck, and CBAM-enhanced attention gates.
        enc_cbam_stages=(False, True, True, True, False),
        dec_cbam_stages=(True, True, True, False),
        gate_use_cbam=True,
        bottleneck_se=True,
        cbam_reduction=16,
        se_reduction=16
    )
    y = model(x)
    print("out:", y.shape)  # expect [2, 4, 256, 256]
