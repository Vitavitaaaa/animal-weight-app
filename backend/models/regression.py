import torch
import torch.nn as nn
import torchvision.models as models

from backend.models.segmentation import UNetEncoder


class TwoStreamRegressor(nn.Module):
    def __init__(self, segmentation_model):
        super().__init__()

        # ================= RGB =================

        self.rgb_backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT
        )

        self.rgb_backbone.fc = nn.Identity()

        # ================= SEG =================

        self.seg_encoder = UNetEncoder(segmentation_model)

        for p in self.seg_encoder.parameters():
            p.requires_grad = False

        self.seg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # ================= HEAD =================

        self.head = nn.Sequential(
            nn.Linear(512 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, img):

        rgb_feat = self.rgb_backbone(img)

        seg_feat = self.seg_encoder(img)
        seg_feat = self.seg_pool(seg_feat)

        x = torch.cat([
            rgb_feat,
            seg_feat
        ], dim=1)

        out = self.head(x)

        return out.squeeze(1)