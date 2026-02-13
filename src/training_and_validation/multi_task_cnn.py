import torch
import torch.nn as nn
import torchvision.models as models

class MultitaskCNN(nn.Module):
    def __init__(self, backbone_name='mobilenet_v2', num_age_classes=1):
        super(MultitaskCNN, self).__init__()
        
        # 1. Load the Backbone (ImageNet Pretrained)
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            self.in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() 
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            self.in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else: # MobileNetV2 (Default)
            self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
            self.in_features = self.backbone.last_channel
            self.backbone.classifier = nn.Identity()

        # 2. Shared bottleneck layer
        self.shared_layer = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # 3. Task-Specific Heads
        self.gender_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )
        
        # AGE HEAD: Uses the exact number of classes in the data
        self.age_head = nn.Linear(256, num_age_classes)

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = torch.mean(features, dim=[2, 3]) # Global Average Pool
            
        shared = self.shared_layer(features)
        gender = self.gender_head(shared)
        age_logits = self.age_head(shared)
        
        return gender, age_logits