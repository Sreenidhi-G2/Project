"""
VeriFace - Phase 2
Model: pretrained EfficientNet-B4 backbone (via timm) fine-tuned for
binary real/fake classification.

Why EfficientNet-B4 and not a custom architecture: this is the standard
choice in the deepfake-detection literature (alongside Xception). Using
a pretrained, well-understood backbone and being explicit about that
choice is more defensible in an interview than claiming architectural
novelty you don't need.
"""

import timm
import torch.nn as nn


class DeepfakeDetector(nn.Module):
    def __init__(self, backbone_name="efficientnet_b4", pretrained=True,
                 freeze_backbone_layers=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),  # single logit - binary classification
        )

        if freeze_backbone_layers:
            self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze roughly the first half of backbone parameters initially.
        Call unfreeze_all() after a few epochs once the classifier head
        has stabilized, per the plan (Day 3: freeze early -> Day 4: unfreeze
        gradually)."""
        params = list(self.backbone.parameters())
        freeze_count = len(params) // 2
        for p in params[:freeze_count]:
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        logit = self.classifier(features)
        return logit.squeeze(1)  # shape: (batch,)