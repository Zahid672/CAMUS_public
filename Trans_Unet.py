import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
#    Core CNN Blocks
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    in_ch is channels AFTER concatenation, i.e., in_ch = ch_up + ch_skip.
    Internally we expect the 'up' input to have (in_ch // 2) channels (ch_up).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        ch_up = in_ch // 2
        self.up = nn.ConvTranspose2d(ch_up, ch_up, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if necessary (for odd sizes)
        diffY = skip.size(-2) - x.size(-2)
        diffX = skip.size(-1) - x.size(-1)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------
#   Transformer Components
# ---------------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # Self-attention
        q = k = v = self.norm1(x)
        x = x + self.proj_drop(self.attn(q, k, v)[0])
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBottleneck(nn.Module):
    """
    Operates on encoder bottleneck feature map (B, C, H', W').
    Flattens to tokens (B, N, E), adds learned positional embeddings,
    runs L encoder layers, reshapes back to (B, C, H', W').
    """
    def __init__(self, in_channels, h, w, depth=4, embed_dim=None, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.embed_dim = embed_dim or in_channels
        self.proj_in = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1, bias=False)

        self.num_tokens = h * w
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim=self.embed_dim, num_heads=num_heads,
                                    mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(depth)
        ])

        self.proj_out = nn.Conv2d(self.embed_dim, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.h and W == self.w, f"Bottleneck expects HxW={self.h}x{self.w}, got {H}x{W}"
        x = self.proj_in(x)                       # (B, E, H, W)
        x = x.flatten(2).transpose(1, 2)          # (B, N, E)
        x = x + self.pos_embed                    # positional embedding

        for blk in self.layers:
            x = blk(x)                            # (B, N, E)

        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)  # (B, E, H, W)
        x = self.proj_out(x)                      # (B, C, H, W)
        return x


# ---------------------------
#     Transformer U-Net
# ---------------------------
class TransUNetLite(nn.Module):
    """
    U-Net with a Transformer bottleneck on the lowest-resolution features.
    Encoder downsamples to H/8, then we pool to H/16 for the Transformer,
    then decode: fuse with x4 (H/8), x3 (H/4), x2 (H/2), x1 (H).
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        img_size=224,
        base_ch=64,
        trans_depth=4,
        trans_heads=8,
        trans_embed_dim=None,  # default to bottleneck C
        trans_mlp_ratio=4.0,
        trans_drop=0.0
    ):
        super().__init__()
        assert img_size % 16 == 0, "img_size must be divisible by 16."

        # -------- Encoder --------
        self.inc   = DoubleConv(in_channels, base_ch)           # 64, H
        self.down1 = Down(base_ch, base_ch * 2)                 # 128, H/2
        self.down2 = Down(base_ch * 2, base_ch * 4)             # 256, H/4
        self.down3 = Down(base_ch * 4, base_ch * 8)             # 512, H/8

        # -------- Bottleneck (pooled to H/16) --------
        H_b = W_b = img_size // 16
        self.bottleneck_conv = DoubleConv(base_ch * 8, base_ch * 8)  # keep 512
        self.trans = TransformerBottleneck(
            in_channels=base_ch * 8,
            h=H_b, w=W_b,
            depth=trans_depth,
            embed_dim=trans_embed_dim or base_ch * 8,
            num_heads=trans_heads,
            mlp_ratio=trans_mlp_ratio,
            drop=trans_drop
        )

        # -------- Decoder --------
        # Up1: b_up (512) + x4 (512) = 1024 -> 256
        self.up1 = Up(in_ch=base_ch * 16, out_ch=base_ch * 4)   # 1024 -> 256
        # Up2: d1 (256) + x3 (256) = 512 -> 128
        self.up2 = Up(in_ch=base_ch * 8,  out_ch=base_ch * 2)   # 512  -> 128
        # Up3: d2 (128) + x2 (128) = 256 -> 64
        self.up3 = Up(in_ch=base_ch * 4,  out_ch=base_ch)       # 256  -> 64
        # Up4: d3 (64) up to H and fuse with x1 (64) -> 128 -> 64
        self.up4 = Up(in_ch=base_ch * 2,  out_ch=base_ch)       # (64+64)->64
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        assert H % 16 == 0 and W % 16 == 0, "Input spatial size must be divisible by 16."

        # -------- Encoder --------
        x1 = self.inc(x)           # (B, 64,  H,   W)
        x2 = self.down1(x1)        # (B, 128, H/2, W/2)
        x3 = self.down2(x2)        # (B, 256, H/4, W/4)
        x4 = self.down3(x3)        # (B, 512, H/8, W/8)

        # Pool to H/16, W/16 for Transformer
        x4_pool = F.max_pool2d(x4, 2)  # (B, 512, H/16, W/16)

        # -------- Bottleneck --------
        b = self.bottleneck_conv(x4_pool)  # (B, 512, H/16, W/16)
        b = self.trans(b)                  # (B, 512, H/16, W/16)

        # Upsample back to H/8 for first fusion
        b_up = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)  # (B,512,H/8,W/8)

        # -------- Decoder (correct fusion order) --------
        d1 = self.up1(b_up, x4)    # fuse with x4 (512) -> (B,256,H/4,W/4)
        d2 = self.up2(d1,  x3)     # fuse with x3 (256) -> (B,128,H/2,W/2)
        d3 = self.up3(d2,  x2)     # fuse with x2 (128) -> (B, 64, H,  W)
        d4 = self.up4(d3,  x1)     # fuse with x1 ( 64) -> (B, 64, H,  W)

        out = self.outc(d4)        # logits: (B, num_classes, H, W)
        return out


# ---------------------------
#        Quick Test
# ---------------------------
if __name__ == "__main__":
    # Example for CAMUS: grayscale, 4 classes, 224x224
    model = TransUNetLite(
        in_channels=1,
        num_classes=4,
        img_size=224,         # must match your dataloader crop/resize
        base_ch=64,
        trans_depth=4,
        trans_heads=8,
        trans_drop=0.1
    )
    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  # (2, 4, 224, 224)
