# AttentionUNet_DS.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- Core Blocks --------------------
class ResidualDoubleConv(nn.Module):
    """Two 3x3 convs with GroupNorm + ReLU and a residual path."""
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.project = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        y = y + self.project(x)
        return self.act(y)


class SE(nn.Module):
    """Squeeze-and-Excitation channel recalibration."""
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(1, ch // r)
        self.fc1 = nn.Conv2d(ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class AttentionGate(nn.Module):
    """
    Additive Attention Gate (AG) with GroupNorm + SE refinement.
    g (decoder) is upsampled to x (encoder) size if needed.
    """
    def __init__(self, F_g, F_l, F_int, groups=8, se_ratio=16):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.GroupNorm(num_groups=min(groups, F_int), num_channels=F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.GroupNorm(num_groups=min(groups, F_int), num_channels=F_int),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.se = SE(F_l, r=se_ratio)

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        att = self.psi(self.W_g(g) + self.W_x(x))     # (B,1,H,W)
        x = x * att                                   # spatial gating
        x = self.se(x)                                # channel gating
        return x


class UpBlock(nn.Module):
    """
    Bilinear upsample + 3x3 conv (with GN) -> concat skip -> residual double conv.
    We expose .up and .conv so the caller can run attention between them.
    """
    def __init__(self, in_ch, skip_ch, out_ch, groups=8):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = ResidualDoubleConv(out_ch + skip_ch, out_ch, groups=groups)


# -------------------- Attention U-Net (Deep Supervision Optional) --------------------
class AttentionUNetDS(nn.Module):
    """
    Attention U-Net with:
      - GroupNorm + residual double convs
      - Dilated bottleneck (padding = dilation)
      - Bilinear upsample path
      - SE-enhanced Attention Gates on all skips
      - Optional deep supervision (two aux heads at ~1/2 and ~1/4 scales)
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        base_ch=64,
        depth=5,
        groups=8,
        use_deep_supervision=True,
        use_dilated_bottleneck=True
    ):
        super().__init__()
        assert 4 <= depth <= 6, "Supported depth: 4..6"

        # Channels per level
        enc_chs = [base_ch * (2 ** i) for i in range(depth)]

        # ----- Encoder -----
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch_in = in_channels
        for i in range(depth):
            self.enc.append(ResidualDoubleConv(ch_in, enc_chs[i], groups=groups))
            ch_in = enc_chs[i]
            if i < depth - 1:
                self.pool.append(nn.MaxPool2d(2))

        # ----- Bottleneck (dilated) -----
        dil = 2 if use_dilated_bottleneck else 1
        pad = dil  # keep shape safe
        bott_ch = enc_chs[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bott_ch, bott_ch, 3, padding=pad, dilation=dil, bias=False),
            nn.GroupNorm(num_groups=min(groups, bott_ch), num_channels=bott_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bott_ch, bott_ch, 3, padding=pad, dilation=dil, bias=False),
            nn.GroupNorm(num_groups=min(groups, bott_ch), num_channels=bott_ch),
            nn.ReLU(inplace=True),
        )

        # ----- Decoder + Attention Gates -----
        self.up = nn.ModuleList()
        self.att = nn.ModuleList()
        dec_in = bott_ch
        for i in range(depth - 2, -1, -1):
            skip_ch = enc_chs[i]
            out_ch  = skip_ch
            # up: dec_in -> out_ch
            self.up.append(UpBlock(dec_in, skip_ch, out_ch, groups=groups))
            # AG now expects F_g == out_ch (post-up), F_l == skip_ch
            self.att.append(AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=max(skip_ch // 2, 1), groups=groups))
            dec_in = out_ch

        # ----- Heads -----
        self.head = nn.Conv2d(enc_chs[0], num_classes, kernel_size=1)

        self.use_deep_supervision = use_deep_supervision
        if use_deep_supervision and depth >= 5:
            self.aux1 = nn.Conv2d(enc_chs[1], num_classes, kernel_size=1)  # ~1/2 scale
            self.aux2 = nn.Conv2d(enc_chs[2], num_classes, kernel_size=1)  # ~1/4 scale
        else:
            self.aux1, self.aux2 = None, None

    def forward(self, x):
        # ----- Encoder -----
        feats = []
        h = x
        for i, enc in enumerate(self.enc):
            h = enc(h)
            feats.append(h)
            if i < len(self.pool):
                h = self.pool[i](h)

        # ----- Bottleneck -----
        h = self.bottleneck(h)

        # ----- Decoder with correct AG gating order -----
        aux_feats = []
        for i in range(len(self.up)):
            skip = feats[-(i + 2)]         # pick corresponding encoder skip

            # 1) upsample+conv first to get the gate feature with expected channels
            h_up = self.up[i].up(h)        # channels == out_ch for this stage

            # 2) gate the skip using h_up (not the pre-up h)
            skip_gated = self.att[i](skip, h_up)

            # 3) concat and finish stage conv
            if h_up.shape[-2:] != skip_gated.shape[-2:]:
                h_up = F.interpolate(h_up, size=skip_gated.shape[-2:], mode='bilinear', align_corners=False)
            h = self.up[i].conv(torch.cat([h_up, skip_gated], dim=1))

            # collect deep supervision features (~1/4 and ~1/2) when depth=5
            if self.use_deep_supervision and self.aux1 is not None and self.aux2 is not None:
                if i == 1:      # ~1/4
                    aux_feats.append(h)
                elif i == 2:    # ~1/2
                    aux_feats.append(h)

        logits = self.head(h)

        if self.use_deep_supervision and len(aux_feats) == 2:
            aux_small = self.aux2(aux_feats[0])  # ~1/4
            aux_big   = self.aux1(aux_feats[1])  # ~1/2
            aux_small = F.interpolate(aux_small, size=logits.shape[-2:], mode='bilinear', align_corners=False)
            aux_big   = F.interpolate(aux_big,   size=logits.shape[-2:], mode='bilinear', align_corners=False)
            return logits, aux_big, aux_small

        return logits


# -------------------- quick self-test --------------------
if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 256)
    model = AttentionUNetDS(
        in_channels=1,
        num_classes=4,
        base_ch=64,
        depth=5,
        groups=8,
        use_deep_supervision=True,
        use_dilated_bottleneck=True
    )
    y = model(x)
    if isinstance(y, tuple):
        main, a1, a2 = y
        print("main:", main.shape, "| aux1:", a1.shape, "| aux2:", a2.shape)
    else:
        print("main:", y.shape)
