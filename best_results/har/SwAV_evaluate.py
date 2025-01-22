import sys
sys.path.append('../../')

import torch
import lightning as L

from models import swav
from data_modules.har_swav import HarDataModule
from sklearn.metrics import accuracy_score

def evaluate_model(model, dataset_dl):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_pred = []
    y_true = []
    # For each batch, compute the predictions and compare with the labels.
    for X, y in dataset_dl:
        # Move the model, data and metric to the GPU if available
        logits = model(X)
        predictions = torch.argmax(logits, dim=1, keepdim=True)
        y_true.append(y)
        y_pred.append(predictions)

    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    # Return a tuple with the number of correct predictions and the total number of predictions
    return accuracy_score(y_true, y_pred)

def report_acc(model, dataset_dl, prefix=""):
    acc = evaluate_model(model, dataset_dl)
    print(prefix + " Acc. = {:0.4f}".format(acc))

### -------------------------------------------------------------------------------

# This function must instantiate and configure the datamodule for the downstream task.
# You must not change this function (Check with the professor if you need to change it).
def build_downstream_datamodule() -> L.LightningDataModule:
    return HarDataModule("../../data/har/", batch_size=60)

# This function must instantiate the downstream model and load its weights
# from checkpoint_filename.
# You might change this code, but must ensure it returns a Lightning model initialized with
# Weights saved by the *_train.py script.
def load_downstream_model(checkpoint_filename) -> L.LightningModule:
    backbone = swav.MLPBackbone(input_size = 6*60, layer_sizes = [768, 128, 64, 32])
    model = swav.SwaVLightningModule(backbone=backbone) # , learning_mode="freeze"
    model.load_state_dict(torch.load(checkpoint_filename)["state_dict"])
    return model

# This function must not be changed. 
def main(SSL_technique_prefix):

    # Load the pretrained model
    downstream_model = load_downstream_model(f'{SSL_technique_prefix}-downstream-model.ckpt')

    # Retrieve the train, validation and test sets.
    downstream_datamodule = build_downstream_datamodule()
    train_dl = downstream_datamodule.train_dataloader()
    val_dl   = downstream_datamodule.val_dataloader()
    test_dl  = downstream_datamodule.test_dataloader()    

    # Compute and report the mIoU metric for each subset
    report_acc(downstream_model, train_dl, prefix="   Training dataset")
    report_acc(downstream_model, val_dl,   prefix=" Validation dataset")
    report_acc(downstream_model, test_dl,  prefix="       Test dataset")

if __name__ == "__main__":
    SSL_technique_prefix = "TFC"
    main(SSL_technique_prefix)