import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=(255, 701)):
        super(SegmentationHead, self).__init__()
        self.img_size = img_size
        # Convolutional layers to process transformer output
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # Assuming x is the output from Vision Transformer, shape [batch_size, 768]
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1, 1)  # Reshape to [batch_size, 768, 1, 1] to prepare for convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)  # Upsample to match input size
        return x

class SegmentationModel(pl.LightningModule):
    def __init__(self, backbone, prediction_head, num_classes=6, lr=0.001, freeze_backbone=False):
        super(SegmentationModel, self).__init__()
        self.backbone = backbone
        self.prediction_head = prediction_head
        self.lr = lr
        self.freeze_backbone = freeze_backbone

        # Optionally freeze the backbone parameters
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        features = self.backbone(x)
        return self.prediction_head(features)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        # Ensure targets are 2D
        targets = targets.squeeze(1).long()  # Remove the channel dimension if it exists

        # Calculate loss
        loss = F.cross_entropy(outputs, targets)

        # Logging loss
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        # Ensure targets are 2D
        targets = targets.squeeze(1).long()  # Remove the channel dimension if it exists

        # Calculate loss
        loss = F.cross_entropy(outputs, targets)

        # Logging loss
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer
