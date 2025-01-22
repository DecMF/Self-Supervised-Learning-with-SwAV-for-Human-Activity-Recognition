import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
import torch.nn.functional as F

from transforms.ts_transforms import augment_time_series, subsample

# SwAV model
class SwaV(nn.Module):
    def __init__(self, backbone, n_prototypes):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(128, 128, 64)
        self.prototypes = SwaVPrototypes(64, n_prototypes=n_prototypes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

# MLP backbone
class MLPBackbone(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()
        layers = []
        for output_size in layer_sizes:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            input_size = output_size
        layers.append(nn.Linear(input_size, 128))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.model(x)

# Custom dataset for multi-feature time series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample, dtype=torch.float32), 0


class SwaVLightningModule(pl.LightningModule):
    def __init__(self, backbone, n_prototypes, lr):
        super().__init__()
        self.model = SwaV(backbone, n_prototypes)
        self.criterion = SwaVLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        views = batch[0].to(self.device)
        self.model.prototypes.normalize()
        high_resolution_views = [views, torch.tensor(augment_time_series(views.cpu().numpy()), dtype=torch.float32).to(self.device)]
        low_resolution_views = [subsample(view.cpu().numpy()) for view in high_resolution_views]
        low_resolution_views = [torch.tensor(view, dtype=torch.float32).to(self.device) for view in low_resolution_views]
        low_resolution_views = [F.interpolate(view.unsqueeze(1), size=(views.size(1), views.size(2)), mode='bilinear').squeeze(1) for view in low_resolution_views]
        all_views = high_resolution_views + low_resolution_views
        all_views = [view.permute(0, 2, 1) for view in all_views]
        multi_crop_features = [self.model(view) for view in all_views]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        views = batch[0].to(self.device)
        self.model.prototypes.normalize()
        high_resolution_views = [views, torch.tensor(augment_time_series(views.cpu().numpy()), dtype=torch.float32).to(self.device)]
        low_resolution_views = [subsample(view.cpu().numpy()) for view in high_resolution_views]
        low_resolution_views = [torch.tensor(view, dtype=torch.float32).to(self.device) for view in low_resolution_views]
        low_resolution_views = [F.interpolate(view.unsqueeze(1), size=(views.size(1), views.size(2)), mode='bilinear').squeeze(1) for view in low_resolution_views]
        all_views = high_resolution_views + low_resolution_views
        all_views = [view.permute(0, 2, 1) for view in all_views]
        multi_crop_features = [self.model(view) for view in all_views]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# Data module for time series data
class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size, num_workers=8):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_size = int(0.8 * len(self.data))
        val_size = len(self.data) - train_size
        self.train_data, self.val_data = random_split(self.data, [train_size, val_size])

    def train_dataloader(self):
        train_dataset = TimeSeriesDataset(self.train_data, transform=augment_time_series)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = TimeSeriesDataset(self.val_data)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

# Load data
path = './data_clean.npy'
data = np.load(path)

# Define configurations
layer_sizes_list = [
    [768, 64, 32, 16],
    [768, 128, 64, 32],
    [768, 256, 128, 64]
]
n_prototypes_list = [16, 32, 64]
lr_list = [0.001, 0.0001]

for layer_sizes in layer_sizes_list:
    for n_prototypes in n_prototypes_list:
        for lr in lr_list:
            input_size = 6 * data.shape[1]
            backbone = MLPBackbone(input_size, layer_sizes)
            model_name = f'MLP_{layer_sizes[1]}_{layer_sizes[2]}_{layer_sizes[3]}_Prototypes_{n_prototypes}_LR_{lr}'
            model = SwaVLightningModule(backbone, n_prototypes, lr)
            datamodule = TimeSeriesDataModule(data, batch_size=128)
            
            # Model checkpoint callback to save the best model
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath='./',
                filename=model_name + '_{epoch:02d}_{val_loss:.2f}',
                save_top_k=1,
                mode='min'
            )
            
            trainer = pl.Trainer(
                max_epochs=5, 
                callbacks=[EarlyStopping(monitor='val_loss', patience=10), checkpoint_callback]
            )
            trainer.fit(model, datamodule)
            
            # Save the backbone model
            torch.save(backbone.state_dict(), f'{model_name}_backbone.pth')


