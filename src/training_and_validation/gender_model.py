import torch
import torch.nn as nn
import torchvision.models as models


class GenderCNN(nn.Module):
    def __init__(self, backbone_name='mobilenet_v2'):
        super(GenderCNN, self).__init__()

        # Load ImageNet pretrained backbone
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:  # Default: MobileNetV2
            self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
            in_features = self.backbone.last_channel
            self.backbone.classifier = nn.Identity()

        # Gender classification head
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)   # Output logits (NO sigmoid here)
        )

    def forward(self, x):
        features = self.backbone(x)

        if len(features.shape) > 2:
            features = torch.mean(features, dim=[2, 3])  # Global Average Pooling

        gender_logits = self.gender_head(features)
        return gender_logits
