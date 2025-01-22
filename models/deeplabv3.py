import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Optional, Sequence

from torchvision.models.resnet import resnet50
from torchvision.models._utils import IntermediateLayerGetter
import types

import lightning as L

from torchvision.models.segmentation import DeepLabV3

class DeepLabV3Model(L.LightningModule):
    def __init__(self, backbone=None, pred_head=None, num_classes=6):
        super().__init__()
        if backbone:
            self.backbone = backbone
        else:
            self.backbone = DeepLabV3Backbone()
        if pred_head:
            self.pred_head = pred_head
        else:
            self.pred_head = DeepLabV3PredictionHead(num_classes=num_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        input_shape = x.shape[-2:]
        h = self.backbone(x)
        z = self.pred_head(h)
        # Upscaling
        return F.interpolate(z, size=input_shape, mode="bilinear", align_corners=False)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X.float())
        # Compute the loss
        loss = self.loss_fn(y_hat, y.squeeze(1).long())
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X, y = batch
        y_hat = self.forward(X.float())
        # Compute the loss
        val_loss = self.loss_fn(y_hat, y.squeeze(1).long())
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001)
        return optimizer

class DeepLabV3Backbone(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        RN50model = resnet50(replace_stride_with_dilation=[False, True, True])
        self.RN50model = RN50model

    def freeze_weights():
        for param in RN50model.parameters():
            param.requires_grad = False

    def unfreeze_weights():
        for param in RN50model.parameters():
            param.requires_grad = True

    def forward(self, x):
            x = self.RN50model.conv1(x)
            x = self.RN50model.bn1(x)
            x = self.RN50model.relu(x)
            x = self.RN50model.maxpool(x)
            x = self.RN50model.layer1(x)
            x = self.RN50model.layer2(x)
            x = self.RN50model.layer3(x)
            x = self.RN50model.layer4(x)
            #x = self.RN50model.avgpool(x)      # These should be removed for deeplabv3
            #x = torch.RN50model.flatten(x, 1)  # These should be removed for deeplabv3
            #x = self.RN50model.fc(x)           # These should be removed for deeplabv3
            return x

class DeepLabV3PredictionHead(nn.Sequential):
    def __init__(self,
                 in_channels: int = 2048,
                 num_classes: int = 6,
                 atrous_rates: Sequence[int] = (12, 24, 36)) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

## The following code was extracted from torchvision.models.segmentation.deeplabv3.py
## TODO: Find a way not to replicate it here
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)
