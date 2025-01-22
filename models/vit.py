import torch
import lightning as L
import torchvision.transforms.v2 as T
from torch import nn
from dataclasses import dataclass
from typing import Callable, Optional, Iterable, Literal, Tuple, TypeAlias
from transformers.modeling_outputs import ModelOutput
from transformers.models.vit import ViTConfig, ViTModel


ImageTransform: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
BatchData = Tuple[torch.Tensor, ...]


def get_image_and_patch_tuple(config: ViTConfig):
    image_size, patch_size = config.image_size, config.patch_size

    if not isinstance(image_size, Iterable):
        image_size = (image_size, image_size)
    if not isinstance(patch_size, Iterable):
        patch_size = (patch_size, patch_size)

    return image_size, patch_size


class ViTPreprocessor(ImageTransform):
    def __init__(self, config: ViTConfig, do_pad: bool = False):
        self.do_pad = do_pad
        image_size, _ = get_image_and_patch_tuple(config)
        self.image_size = image_size

    def _pad(self, im: torch.Tensor) -> torch.Tensor:
        h, w = im.shape[-2:]
        H, W = self.image_size
        pad_h = max(0, H - h)
        pad_w = max(0, W - w)
        if pad_h > 0 or pad_w > 0:
            return T.functional.pad(im, [0, 0, pad_w, pad_h], padding_mode="reflect")
        return im

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        # Normalize images in batch
        t = T.functional.to_grayscale(t, 1)
        t = T.functional.autocontrast(t)
        if self.do_pad:
            return self._pad(t)
        return t


@dataclass
class ViTEncoderOutput(ModelOutput):
    sequence_output: torch.Tensor
    class_embedding_output: Optional[torch.Tensor] = None


class ViTEncoder(nn.Module):
    """
        Transformer masked encoder

        Params
        ------
            config: ViTConfig
                Transformers config
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.vit = ViTModel(
            config, add_pooling_layer=False, use_mask_token=True
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_class_embedding: Optional[bool] = None
    ) -> ViTEncoderOutput:
        # Forward model
        outputs = self.vit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            output_attentions=output_attentions,
            return_dict=True
        )
        sequence_outputs = outputs.last_hidden_state
        if output_class_embedding:
            return ViTEncoderOutput(
                sequence_output=sequence_outputs[:, 1:],
                class_embedding_output=sequence_outputs[:, 0, :]
            )
        return ViTEncoderOutput(
            sequence_output=sequence_outputs[:, 1:]
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class ViTPooling(nn.Module):
    """
        Lightweight embedding generator from encoder outputs
    """

    def __init__(
        self,
        *pooling_modes: Literal['cls', 'mean']
    ) -> None:
        super().__init__()
        self.pooling_modes = pooling_modes if pooling_modes else ('cls',)

    def forward(self, output: ViTEncoderOutput) -> torch.Tensor:
        embedding = []
        if 'cls' in self.pooling_modes and output.class_embedding_output is not None:
            embedding.append(output.class_embedding_output)
        if 'mean' in self.pooling_modes:
            batch_size, sequence_len, hidden_size = output.sequence_output.shape
            sequence_embedding = output.sequence_output.sum(dim=1)
            embedding.append(sequence_embedding/sequence_len)

        return torch.concat(embedding, dim=1)


class ViTConvDecoder(nn.Module):
    """
        Lightweight decoder using a single deconvolution layer

        Params
        ------
            config: ViTConfig
                config used in encoder
            out_channels: Optional[int]
                output channels to segmentation tasks
                default is config.num_channels
    """

    def __init__(self, config: ViTConfig, out_channels: int = None) -> None:
        super().__init__()
        image_size, patch_size = get_image_and_patch_tuple(config)
        self.height = image_size[0]//patch_size[0]
        self.width = image_size[1]//patch_size[1]

        if out_channels is None:
            out_channels = config.num_channels

        self.deconv = nn.ConvTranspose2d(
            in_channels=config.hidden_size,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # Reshape to (batch_size, num_channels, height, width)
        batch_size, _, num_channels = sequence_output.shape
        sequence_output = (
            sequence_output.permute(0, 2, 1)
            .reshape(batch_size, num_channels, self.height, self.width)
        )
        return self.deconv(sequence_output)


class ViTForImageSegmentation(L.LightningModule):
    def __init__(
        self,
        config: ViTConfig,
        prediction_head: Optional[nn.Module] = None,
        num_classes: int = 6
    ) -> None:
        super().__init__()
        self.preprocessor = ViTPreprocessor(config, do_pad=True)
        self.backbone = ViTEncoder(config)
        if prediction_head is not None:
            self.prediction_head = prediction_head
        else:
            self.prediction_head = ViTConvDecoder(config, num_classes)
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        x = self.pre_process(x)
        z = self.backbone(x)
        y = self.prediction_head(z[0])
        return self.post_process(y, input_shape)

    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        return self.preprocessor(x)

    def post_process(
        self,
        pred: torch.Tensor,
        expected_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
            Adjust prediction shape to original shape
        """
        H, W = pred.shape[-2:]
        h, w = expected_shape
        if H > h or W > w:
            return T.functional.crop(pred, 0, 0, h, w)
        return pred

    def step(self, batch: BatchData) -> torch.Tensor:
        X, y = batch
        y_pred = self.forward(X)
        return self.loss_fn(y_pred, y.squeeze(1).long())

    def training_step(self, batch: BatchData, batch_idx) -> torch.Tensor:
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: BatchData, batch_idx) -> torch.Tensor:
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss
