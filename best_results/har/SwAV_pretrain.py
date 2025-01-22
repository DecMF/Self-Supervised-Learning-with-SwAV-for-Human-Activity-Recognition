import sys

sys.path.append("../../")

from torch import nn
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models import swav
from data_modules.har_swav import SWAVDataModule


### -------------------------------------------------------------------------------


# This function must save the weights of the pretrained model
def pretext_save_backbone_weights(pretext_model, checkpoint_filename):
    print(f"Saving backbone pretrained weights at {checkpoint_filename}")
    torch.save(pretext_model.backbone.state_dict(), checkpoint_filename)


# This function must instantiate and configure the datamodule for the pretext task
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure it returns a Lightning DataModule.
def build_pretext_datamodule() -> L.LightningDataModule:
    return SWAVDataModule(
        "../../data/pretext_har/",
        batch_size=128,
        drop_last = True
    )



# This function must instantiate and configure the pretext model
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure it returns a Lightning model.
def build_pretext_model() -> L.LightningModule:
    # Build the backbone
    backbone = swav.MLPBackbone(input_size = 6*60, layer_sizes = [768, 128, 64, 32])
    model = swav.SwaVLightningModule(backbone=backbone)
    return model


# This function must instantiate and configure the lightning trainer
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure you return a Lightning trainer.
def build_lightning_trainer() -> L.Trainer:
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, verbose=False, mode="min"),
        ModelCheckpoint(
            monitor="val_loss",
            filename="SWAV-model",
            mode="min",
        ),
    ]
    return L.Trainer(
        accelerator="cpu",
        max_epochs=3,
        logger=False,
        callbacks=callbacks,
        val_check_interval=0.5,
        # max_steps=30,  # for debugging
    )


# This function must not be changed.
def main(SSL_technique_prefix):

    # Build the pretext model, the pretext datamodule, and the trainer
    pretext_model = build_pretext_model()
    pretext_datamodule = build_pretext_datamodule()
    lightning_trainer = build_lightning_trainer()

    # Fit the pretext model using the pretext_datamodule
    lightning_trainer.fit(pretext_model, pretext_datamodule)

    # Save the backbone weights
    output_filename = f"./{SSL_technique_prefix}_pretrained_backbone_weights.pth"
    pretext_save_backbone_weights(pretext_model, output_filename)


if __name__ == "__main__":
    SSL_technique_prefix = "SWAV"
    main(SSL_technique_prefix)