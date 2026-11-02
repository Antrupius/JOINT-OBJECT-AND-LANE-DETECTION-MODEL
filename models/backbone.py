"""Compact backbones returning stride-8, stride-16 and stride-32 feature maps."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, g=1, act=nn.GELU):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, k, s, k // 2, groups=g, bias=False)
        self.b = nn.BatchNorm2d(cout)
        self.a = act()

    def forward(self, x): return self.a(self.b(self.c(x)))


class SEModule(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // r, 8), 1),
                                nn.GELU(), nn.Conv2d(max(c // r, 8), c, 1), nn.Sigmoid())

    def forward(self, x): return x * self.fc(x)


class RepViTBlock(nn.Module):
    """Channel mixer -> token mixer (DW) -> channel mixer; SE in odd blocks only.
    Follows RepViT's CNN-port of efficient-ViT design principles."""

    def __init__(self, inp, hidden, oup, se=False):
        super().__init__()
        self.cm1 = nn.Sequential(ConvBnAct(inp, hidden, 1, 1), ConvBnAct(hidden, hidden, 1, 1))
        self.tm = nn.Sequential(ConvBnAct(hidden, hidden, 3, 1, g=hidden), nn.GELU())
        self.se = SEModule(hidden) if se else nn.Identity()
        self.cm2 = ConvBnAct(hidden, oup, 1, 1)
        self.res = nn.Identity() if inp == oup else None

    def forward(self, x):
        r = x if self.res is not None else None
        x = self.cm1(x); x = self.tm(x); x = self.se(x); x = self.cm2(x)
        return x + r if r is not None else x


class RepViT(nn.Module):
    def __init__(self, depths=(2, 2, 4, 2), dims=(32, 64, 128, 256), width_mult=1.0, img_h=384):
        super().__init__()
        dims = [max(8, int(d * width_mult)) for d in dims]
        self.H8, self.H16, self.H32 = img_h // 8, img_h // 16, img_h // 32
        stages, cur = [], dims[0] // 2
        self.stem = ConvBnAct(3, dims[0] // 2, 3, 2)
        self.out_idx = []                       # indices of the block-stacks whose outputs feed the heads
        for si, (d, n) in enumerate(zip(dims, depths)):
            if cur != d:
                stages.append(ConvBnAct(cur, d, 3, 2)); cur = d
            blocks = [RepViTBlock(cur, 2 * d, d, se=(bi % 2 == 1)) for bi in range(n)]
            stages.append(nn.Sequential(*blocks))
            if si >= 1:                         # stride 8, 16, 32 stages
                self.out_idx.append(len(stages) - 1)
        self.stages = nn.ModuleList(stages)
        self.out_dims = [dims[1], dims[2], dims[3]]

    def forward(self, x):
        # FIX: take each scale's feature AFTER that stage's blocks (the old code took it
        # right after the stride-2 conv, so the last stage's blocks never reached any head).
        x = self.stem(x)
        outs = []
        for i, s in enumerate(self.stages):
            x = s(x)
            if i in self.out_idx:
                outs.append(x)
        return outs


def _target_h(h):
    return {h}


def build_backbone(name: str, img_h: int = 384, depth_mult: float = 1.0, width_mult: float = 1.0) -> nn.Module:
    if name == "repvit":
        net = RepViT(width_mult=width_mult)
        net.H8, net.H16, net.H32 = img_h // 8, img_h // 16, img_h // 32
        return net
    if name == "mobilenetv3":
        net = MobileNetV3Lite(width_mult=width_mult)
        net.H8, net.H16, net.H32 = img_h // 8, img_h // 16, img_h // 32
        return net
    raise ValueError(name)

class MobileNetV3Lite(nn.Module):
    """Compact MobileNetV3-style CNN (inverted residuals + SE + h-swish).
    Returns [f_s8, f_s16, f_s32]: the LAST block at each stride (fixes the old `outs[-3:]`,
    which could return three maps from the same resolution)."""
    def __init__(self, width_mult=1.0, img_h=384):
        super().__init__()
        self.H8, self.H16, self.H32 = img_h // 8, img_h // 16, img_h // 32
        def d(c): return max(8, int(c * width_mult))
        conf = [
            # expanded width, out channels, repeats, stride, se, hswish
            (16, 16, 1, 1, False, False), (64, 24, 2, 2, False, False),   # -> stride 4
            (72, 24, 1, 1, False, False), (72, 40, 2, 2, True, False),    # -> stride 8
            (120, 40, 1, 1, True, False), (120, 80, 3, 2, True, True),    # -> stride 16
            (240, 96, 2, 1, True, True), (240, 112, 2, 2, True, True),    # -> stride 32
        ]
        layers, cur, stride_total = [], d(16), 2
        self.stem = ConvBnAct(3, cur, 3, 2, act=_HSwish)
        last_at = {}                                    # stride -> (block index, channels)
        for expand, ch, reps, stride, se, hs in conf:
            ch = d(ch); expand = d(expand)
            for ri in range(reps):
                s = stride if ri == 0 else 1
                layers.append(_InvertedResidual(cur, ch, expand, s, se, hs))
                stride_total *= s
                cur = ch
                last_at[stride_total] = (len(layers) - 1, ch)
        self.blocks = nn.ModuleList(layers)
        self.out_idx = [last_at[8][0], last_at[16][0], last_at[32][0]]
        self.out_dims = [last_at[8][1], last_at[16][1], last_at[32][1]]

    def forward(self, x):
        x = self.stem(x); outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_idx:
                outs.append(x)
        return outs


class _HSwish(nn.Hardswish):                            # real h-swish (old code used x*sigmoid(x))
    pass


class _InvertedResidual(nn.Module):
    def __init__(self, cin, cout, expand, stride, se, hs):
        super().__init__()
        self.use_res = stride == 1 and cin == cout
        act = _HSwish if hs else nn.ReLU
        hid = expand                                    # FIX: was cin * expand (thousands of channels)
        self.conv = nn.Sequential(
            ConvBnAct(cin, hid, 1, 1, act=act) if hid != cin else nn.Identity(),
            ConvBnAct(hid, hid, 3, stride, g=hid, act=act),
            SEModule(hid) if se else nn.Identity(),
            ConvBnAct(hid, cout, 1, 1, act=nn.Identity),
        )
        self.out_channels = cout

    def forward(self, x):
        r = x if self.use_res else None
        x = self.conv(x)
        return x + r if r is not None else x
