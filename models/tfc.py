import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from models.encoder_ts import EncoderConv, EnconderMLP
import torch.fft as fft


class TFCBackbone(nn.Module):
    def __init__(self, input_dim, encoding_dim, seq_len, layer_type):
        super().__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.seq_len = seq_len
        self.layer_type = layer_type

        if layer_type == "conv1d":
            self.encoder_t = EncoderConv(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                h_dims=[self.encoding_dim // 4, self.encoding_dim // 4, self.encoding_dim],
            )

            self.encoder_f = EncoderConv(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                h_dims=[self.encoding_dim // 4, self.encoding_dim // 4, self.encoding_dim],
            )
        elif layer_type == "mlp":
            self.encoder_t = EnconderMLP(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                h_dims=[self.encoding_dim // 4, self.encoding_dim // 2, self.encoding_dim],
            )
            self.encoder_f = EnconderMLP(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                h_dims=[self.encoding_dim // 4, self.encoding_dim // 2, self.encoding_dim],
            )

        encoding_dim_ = self.encoder_t.encoding_dim

        self.projector_t = nn.Sequential(
            nn.Linear(encoding_dim_, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2),
        )
        self.projector_f = nn.Sequential(
            nn.Linear(encoding_dim_, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2),
        )

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


class TFC(L.LightningModule):
    def __init__(
        self,
        backbone,
        loss_fn=None,
    ):
        super().__init__()
        self.backbone = backbone
        if loss_fn is None:
            self.loss_fn = NTXentLoss(0.2, True)
        else:
            self.loss_fn = loss_fn

    def forward(self, x_in_t, x_in_f):
        h_time, z_time, h_freq, z_freq = self.backbone(x_in_t, x_in_f)

        return h_time, z_time, h_freq, z_freq

    def training_step(self, batch, batch_idx):
        data, _, aug1, data_f, aug1_f = batch

        # embeddings
        h_t, z_t, h_f, z_f = self.forward(data, data_f)
        h_t_aug, _, h_f_aug, _ = self.forward(aug1, aug1_f)

        # jointly losses
        loss_t = self.loss_fn(h_t, h_t_aug)
        loss_f = self.loss_fn(h_f, h_f_aug)
        l_TF = self.loss_fn(z_t, z_f)  # this is the initial version of TF loss

        lam = 0.2
        loss = lam * (loss_t + loss_f) + l_TF
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, _, aug1, data_f, aug1_f = batch
        # embeddings
        h_t, z_t, h_f, z_f = self.forward(data, data_f)
        h_t_aug, _, h_f_aug, _ = self.forward(aug1, aug1_f)

        # jointly losses
        loss_t = self.loss_fn(h_t, h_t_aug)
        loss_f = self.loss_fn(h_f, h_f_aug)
        l_TF = self.loss_fn(z_t, z_f)  # this is the initial version of TF loss

        lam = 0.2
        loss = lam * (loss_t + loss_f) + l_TF
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
        return optim


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask(32).type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        # get the device from the input
        device = zis.device
        batch_size = zis.size(0)
        if self.mask_samples_from_same_repr.size(0) != 2 * batch_size:
            self.mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        self.mask_samples_from_same_repr = self.mask_samples_from_same_repr.to(device)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * batch_size, -1
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * batch_size).to(device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


class TFCModel(L.LightningModule):
    def __init__(
        self, backbone, pred_head=None, num_classes=6, learning_mode="finetune"
    ):
        super().__init__()
        self.backbone = backbone
        self.encoding_dim = backbone.encoding_dim
        self.learning_mode = learning_mode
        if pred_head:
            self.pred_head = pred_head
        else:
            self.pred_head = nn.Sequential(
                nn.Linear(self.encoding_dim, self.encoding_dim // 2),
                nn.ReLU(),
                nn.Linear(self.encoding_dim // 2, num_classes)
            )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x_in_t):
        x_in_f = fft.fft(x_in_t).abs()
        _, z_t, _, z_f = self.backbone(x_in_t, x_in_f)
        z = torch.cat((z_t, z_f), dim=1)
        y_hat = self.pred_head(z)
        return y_hat

    def training_step(self, batch, batch_idx):
        data, y = batch
        y_hat = self.forward(data)
        loss = self.loss_fn(y_hat, y.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, y = batch
        y_hat = self.forward(data)
        loss = self.loss_fn(y_hat, y.long())
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        if self.learning_mode == "freeze":
            optimizer = torch.optim.Adam(params=self.pred_head.parameters(), lr=1e-4, weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-4, weight_decay=1e-4)
        return optimizer
