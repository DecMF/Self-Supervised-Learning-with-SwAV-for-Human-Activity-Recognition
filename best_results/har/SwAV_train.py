import sys
sys.path.append('../../')

import torch
import torch.nn as nn
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint
from data_modules.har_swav import HarDataModule
from models import swav

# This function should load the backbone weights
def load_pretrained_backbone(pretrained_backbone_checkpoint_filename):
#    loaded_model = MyModel()`
#    loaded_model.model = AutoModel.from_pretrained("path/to/save/model")

    
    backbone = swav.MLPBackbone(input_size = 6*60, layer_sizes = [768, 128, 64, 32])
    backbone.load_state_dict(torch.load(pretrained_backbone_checkpoint_filename)) 
    return backbone

# This function must instantiate and configure the datamodule for the downstream task.
# You must not change this function (Check with the professor if you need to change it).
def build_downstream_datamodule() -> L.LightningDataModule:
    return HarDataModule("../../data/har/", batch_size=60)

# This function must instantiate and configure the downstream model
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure it returns a Lightning model.
def build_downstream_model(backbone) -> L.LightningModule:
    return swav.SwaVLightningModule(backbone=backbone) # , learning_mode="freeze"

# This function must instantiate and configure the lightning trainer
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure you return a Lightning trainer.
def build_lightning_trainer(SSL_technique_prefix) -> L.Trainer:
    # Configure the ModelCheckpoint object to save the best model 
    # according to validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'./',
        filename=f'{SSL_technique_prefix}-downstream-model',
        save_top_k=1,
        mode='min',
    )
    return L.Trainer(
        accelerator="gpu",
        max_epochs=3,
        #max_steps=10,
        logger=False,
        val_check_interval=0.05,
        callbacks=[checkpoint_callback]
        )

# This function must not be changed. 
def main(SSL_technique_prefix):

    # Load the pretrained backbone
    pretrained_backbone_checkpoint_filename = f"./{SSL_technique_prefix}_pretrained_backbone_weights.pth"
    backbone = load_pretrained_backbone(pretrained_backbone_checkpoint_filename)

    # Build the downstream model, the downstream datamodule, and the trainer
    downstream_model = build_downstream_model(backbone)
    downstream_datamodule = build_downstream_datamodule()
    lightning_trainer = build_lightning_trainer(SSL_technique_prefix)

    # Fit the pretext model using the pretext_datamodule
    lightning_trainer.fit(downstream_model, downstream_datamodule)

    # Save the downstream model    
    #output_filename = f"./{SSL_technique_prefix}_downstream_model.pth"
    #pretext_save_backbone_weights(pretext_model, output_filename)
    #print(f"Pretrained weights saved at: {output_filename}")

if __name__ == "__main__":
    SSL_technique_prefix = "SWAV"
    main(SSL_technique_prefix)