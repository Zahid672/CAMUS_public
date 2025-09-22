# Modified_UNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Blocks --------------------
class ConvGNReLU(nn.Module):
    """
    Two 3x3 convs with GroupNorm + ReLU, optional residual connection.
    Uses padding = dilation to preserve HxW even when dilation > 1.
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, groups=8, dropout=0.0, residual=False, dilation=1):
        super().__init__()
        pad = dilation  # <-- key fix: keep spatial size
        self.residual = residual and (in_ch == out_ch)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad, bias=False, dilation=dilation),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=k, stride=1, padding=pad, bias=False, dilation=dilation),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        y = self.block(x)
        return y + x if self.residual else y


class UpBlock(nn.Module):
    """
    Transposed-conv upsample + skip concat + ConvGNReLU (residual).
    """
    def __init__(self, in_ch, skip_ch, out_ch, groups=8, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvGNReLU(out_ch + skip_ch, out_ch, groups=groups, dropout=dropout, residual=True)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -------------------- UNet --------------------
class UNet(nn.Module):
    """
    - depth:    4–6 (5 recommended for CAMUS 256x256)
    - GroupNorm (stable for small batch)
    - Residual double convs
    - Optional dropout in decoder/bottleneck
    - Optional dilated bottleneck to increase receptive field without extra pooling
    - Optional deep supervision (two auxiliary heads)
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        base_ch=64,
        depth=5,                 # 5 is a sweet spot for CAMUS
        groups=8,
        dropout=0.1,
        deep_supervision=True,
        dilated_bottleneck=True
    ):
        super().__init__()
        assert 4 <= depth <= 6, "Supported depth: 4..6"

        # encoder channels per stage
        chs = [base_ch * (2 ** i) for i in range(depth)]

        # Encoder
        self.enc = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            self.enc.append(ConvGNReLU(in_ch, chs[i], residual=True, groups=groups, dropout=0.0))
            in_ch = chs[i]
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))

        # Bottleneck (dilated convs preserve size due to padding=dilation)
        self.bottleneck = ConvGNReLU(
            chs[-1], chs[-1],
            residual=True,
            groups=groups,
            dropout=dropout,
            dilation=2 if dilated_bottleneck else 1
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        dec_in = chs[-1]
        for i in range(depth - 2, -1, -1):
            self.up_blocks.append(UpBlock(dec_in, chs[i], chs[i], groups=groups, dropout=dropout))
            dec_in = chs[i]

        # Heads
        self.head = nn.Conv2d(chs[0], num_classes, kernel_size=1)

        # Deep supervision (two aux heads roughly at 1/4 and 1/2 scale)
        self.deep_supervision = deep_supervision
        if deep_supervision and depth >= 5:
            self.aux2 = nn.Conv2d(chs[2], num_classes, kernel_size=1)  # ~1/4
            self.aux1 = nn.Conv2d(chs[1], num_classes, kernel_size=1)  # ~1/2
        else:
            self.aux2 = None
            self.aux1 = None

    def forward(self, x):
        # Encoder
        feats = []
        h = x
        for i, enc in enumerate(self.enc):
            h = enc(h)
            feats.append(h)
            if i < len(self.pools):
                h = self.pools[i](h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        aux_logits_feats = []
        for i, up in enumerate(self.up_blocks):
            skip = feats[-(i + 2)]  # walk back encoder feats
            h = up(h, skip)

            # Keep features from mid decoder stages for deep supervision
            # We pick the last two (closer to full resolution)
            if self.deep_supervision and len(self.up_blocks) - i in (3, 2):
                aux_logits_feats.append(h)

        logits = self.head(h)

        # Return aux heads if enabled and available
        if self.deep_supervision and self.aux1 is not None and self.aux2 is not None and len(aux_logits_feats) >= 2:
            aux_big  = F.interpolate(self.aux1(aux_logits_feats[-1]), size=logits.shape[-2:], mode='bilinear', align_corners=False)
            aux_small= F.interpolate(self.aux2(aux_logits_feats[-2]), size=logits.shape[-2:], mode='bilinear', align_corners=False)
            return logits, aux_big, aux_small

        return logits


# -------------------- quick self-test --------------------
if __name__ == "__main__":
    # CAMUS uses 1-channel input; 256x256 recommended
    x = torch.randn(2, 1, 256, 256)
    model = UNet(
        in_channels=1,
        num_classes=4,
        base_ch=64,
        depth=5,                 # keep 5 for CAMUS
        groups=8,
        dropout=0.1,
        deep_supervision=True,
        dilated_bottleneck=True  # increase RF without changing HxW
    )
    out = model(x)
    if isinstance(out, tuple):
        y, a1, a2 = out
        print("main:", y.shape, "| aux1:", a1.shape, "| aux2:", a2.shape)  # expect [B,4,256,256] each
    else:
        print("main:", out.shape)
