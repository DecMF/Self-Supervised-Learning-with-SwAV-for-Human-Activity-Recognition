import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import Adam, SGD

class DINOv2(pl.LightningModule):
    def __init__(self, backbone, teacher_backbone=None, projection_head=None, out_dim=128, lr=0.001, update_rate=0.9):
        super().__init__()
        self.backbone = backbone
        self.teacher_backbone = teacher_backbone
        self.projection_head = projection_head
        self.loss_fn = DINOLoss(out_dim=out_dim)
        self.lr = lr
        self.update_rate = update_rate
        self.register_buffer("teacher_params", None)

    def forward(self, x):
        x = self.backbone(x)
        if self.projection_head is not None:
            x = self.projection_head(x)
        return x

    def teacher_forward(self, x):
        if self.teacher_backbone is None:
            raise ValueError("Teacher backbone is not provided.")
        x = self.teacher_backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        original, student_transform, teacher_transform = batch

        # Forward pass through student model
        student_output = self(student_transform)

        # Forward pass through teacher model
        with torch.no_grad():
            teacher_output = self.teacher_forward(teacher_transform)

        # Compute loss
        loss = self.loss_fn(student_output, teacher_output)

        # Logging loss
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr)
        return optimizer

    def on_epoch_start(self):
        print(f"Epoch {self.current_epoch}")

    def on_epoch_end(self):
        print(f"Epoch {self.current_epoch} ended")

    def on_train_batch_start(self, batch, batch_idx):
        print(f"Training batch {batch_idx}")

    def on_validation_batch_start(self, batch, batch_idx):
        print(f"Validation batch {batch_idx}")

    def on_batch_end(self):
        # Update teacher model with EMA parameters based on student model
        if self.teacher_params is None:
            self.teacher_params = {k: v.clone().detach() for k, v in self.backbone.state_dict().items()}
        else:
            for k, v in self.backbone.state_dict().items():
                self.teacher_params[k] = self.teacher_params[k] * self.update_rate + v * (1 - self.update_rate)

class DINOLoss(pl.LightningModule):
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output):
        self.apply_center_update()
        # Teacher centering and sharpening
        teacher_output_centered = (teacher_output - self.center) / self.teacher_temp
        teacher_output_softmaxed_centered = F.softmax(teacher_output_centered, dim=-1)
        return teacher_output_softmaxed_centered

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        teacher_output_softmaxed_centered = self.softmax_center_teacher(teacher_output)

        student_log_softmax = F.log_softmax(student_output / self.student_temp, dim=-1)

        loss = torch.sum(-teacher_output_softmaxed_centered * student_log_softmax, dim=-1)
        loss = loss.mean()

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        self.reduce_handle = self.async_batch_center.clone().detach()
        self.reduce_handle.requires_grad = False

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            self.async_batch_center = self.async_batch_center.clone().detach()
            self.center = self.center * self.center_momentum + self.async_batch_center * (1 - self.center_momentum)
            self.updated = True
