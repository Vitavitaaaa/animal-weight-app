import torch
import torch.nn as nn
import numpy as np

COLOR_MAP_B2 = {
    (255,240,0): 0,   # sticker
    (0,255,0): 1,     # cattle
    (0,0,255): 2,     # background
    (255,0,0): 3      # ground
}

COLOR_MAP_B3 = {
    (0,117,255): 0,
    (255,30,249): 1,
    (0,255,193): 2
}



NUM_CLASSES = 4


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = C(3, 32)
        self.enc2 = C(32, 64)
        self.enc3 = C(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        self.dec1 = C(128 + 64, 64)
        self.dec2 = C(64 + 32, 32)

        self.out = nn.Conv2d(32, NUM_CLASSES, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d1 = self.up(e3)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        mask_logits = self.out(d2)

        return mask_logits


class UNetEncoder(nn.Module):
    def __init__(self, unet):
        super().__init__()

        self.enc1 = unet.enc1
        self.enc2 = unet.enc2
        self.enc3 = unet.enc3

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        return e3