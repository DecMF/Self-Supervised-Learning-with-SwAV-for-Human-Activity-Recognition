import sys

sys.path.append("../../")

import os
from matplotlib import pyplot as plt
import pandas as pd
from torch import nn
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import models.tfc as tfc
from data_modules.har import TFCDataModule, HarDataModule
import torch.fft as fft


def build_datamodule(mode: str = "pretext", frac: int = 1) -> L.LightningDataModule:
    """Builds the datamodule for the pretext or downstream task."""
    if mode == "pretext":
        return TFCDataModule("../../data/pretext_har/", batch_size=128, drop_last=True)
    elif mode == "downstream":
        return HarDataModule("../../data/har/", batch_size=10, frac=frac)


def build_pretext_model(encoding_dim, layer_type) -> L.LightningModule:
    """Builds the pretext model with the given parameters."""
    backbone = tfc.TFCBackbone(
        input_dim=6,
        encoding_dim=encoding_dim,
        seq_len=60,
        layer_type=layer_type,
    )
    backbone = tfc.TFC(backbone)
    return backbone


def build_lightning_trainer(logging_name: str = "model") -> L.Trainer:
    """Builds the lightning trainer with the given parameters."""
    if os.path.exists(f"lightning_logs/{logging_name}"):
        os.system(f"rm -r lightning_logs/{logging_name}")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min", min_delta = 1e-4),
        ModelCheckpoint(
            monitor="val_loss",
            filename="TFC-model",
            mode="min",
        ),
    ]
    logger = CSVLogger("lightning_logs", name=logging_name)
    return L.Trainer(
        accelerator="gpu",
        max_epochs=100,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=0.5,
        # max_steps=30,  # for debugging
    )


def get_embeddings(backbone, data_loader):
    backbone.eval()
    embeddings = []
    y = []
    for batch in data_loader:
        if len(batch) == 2:
            data, y_ = batch
            data_f = fft.fft(data).abs()
        else:
            data, y_, _, data_f, _ = batch
        h_t, z_t, h_f, z_f = backbone(data, data_f)
        z = torch.cat([z_t, z_f], dim=1)
        embeddings.append(z)
        y.append(y_)
    embeddings = torch.cat(embeddings, dim=0).detach().cpu().numpy()
    y = torch.cat(y, dim=0).detach().cpu().numpy()
    return embeddings, y


def plot_embeddings(embeddings, y, figname, title=""):
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(4, 4))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap="tab10")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

    return


def eval_linear_model(model, datamodule):
    """Calculates train, validation and test accuracy of the model using a linear classifier.


    Parameters
    ----------
    model : TFCModel
        Fitted model to use the backbone.
    datamodule : TFCDataModule
        DataModule to use for the evaluation.

    Returns
    -------
    train_acc : float
        Train accuracy of the model.
    val_acc : float
        Validation accuracy of the model.
    test_acc : float
        Test accuracy of the model.
    """
    model.eval()

    def get_acc(model, dl):
        y_pred = []
        y_ground = []
        for batch in dl:
            x, y = batch
            logits = model(x)
            predictions = torch.argmax(logits, dim=1, keepdim=True)
            y_pred.append(predictions)
            y_ground.append(y)

        y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
        y_ground = torch.cat(y_ground, dim=0).detach().cpu().numpy()
        return accuracy_score(y_ground, y_pred)

    train_acc = get_acc(model, datamodule.train_dataloader())
    val_acc = get_acc(model, datamodule.val_dataloader())
    test_acc = get_acc(model, datamodule.test_dataloader())

    return train_acc, val_acc, test_acc


def eval_knn(model, datamodule):
    """Calculates train, validation and test accuracy of the model using KNN with k=2.

    Parameters
    ----------
    model : TFCModel
        Fitted model to use the backbone.
    datamodule : TFCDataModule
        DataModule to use for the evaluation.

    Returns
    -------
    train_acc : float
        Train accuracy of the model.
    val_acc : float
        Validation accuracy of the model.
    test_acc : float
        Test accuracy of the model.
    """
    embeddings, y = get_embeddings(model.backbone, datamodule.train_dataloader())
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(embeddings, y)
    train_acc = accuracy_score(y, knn.predict(embeddings))

    embeddings, y = get_embeddings(model.backbone, datamodule.val_dataloader())
    val_acc = accuracy_score(y, knn.predict(embeddings))

    embeddings, y = get_embeddings(model.backbone, datamodule.test_dataloader())
    test_acc = accuracy_score(y, knn.predict(embeddings))
    return train_acc, val_acc, test_acc


def plot_train_val(path, title, figname):
    df = pd.read_csv(path + "metrics.csv")
    if "train_loss_step" in df.columns:
        train_loss_column = "train_loss_step"
    else:
        train_loss_column = "train_loss_epoch"
        
    train_loss = df[["step", train_loss_column]].copy().dropna()
    val_loss = df[["step", "val_loss"]].copy().dropna()

    plt.figure(figsize = (4, 4))
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.plot(train_loss.step, train_loss[train_loss_column], label = "Train loss")
    plt.plot(val_loss.step, val_loss.val_loss, label = "Val loss")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def evaluate_model(downstream_model, downstream_datamodule, results, encoding_dim, layer_type, learning_mode):
    train_acc, validation_acc, test_acc = eval_linear_model(
        downstream_model, downstream_datamodule
    )
    train_acc_knn, validation_acc_knn, test_acc_knn = eval_knn(
        downstream_model, downstream_datamodule
    )
    embeddings, y = get_embeddings(
        downstream_model.backbone, downstream_datamodule.train_dataloader()
    )
    plot_embeddings(
        embeddings,
        y,
        f"figures/tsne_{encoding_dim}_{layer_type}_{learning_mode}.png",
        f"t-SNE of embeddings\nconfig:{encoding_dim}_{layer_type}_{learning_mode}\nval.acc.:{validation_acc:.2f}",
    )

    plot_train_val(
        f"lightning_logs/TFC_downstream_{learning_mode}_{encoding_dim}_{layer_type}/version_0/",
        f"Training and validation loss\nconfig:{encoding_dim}_{layer_type}_{learning_mode}",
        f"figures/train_val_{encoding_dim}_{layer_type}_{learning_mode}.png",
    )
    results.append(
        {
            "encoding_dim": encoding_dim,
            "layer_type": layer_type,
            "learning_mode": learning_mode,
            "train_acc": train_acc,
            "validation_acc": validation_acc,
            "train_acc_knn": train_acc_knn,
            "validation_acc_knn": validation_acc_knn,
        }
    )
    return results


def fit_backbones(ENCODING_DIMS):
    LAYERS_TYPES = ["conv1d", "mlp"]

    combinations = [
        (encoding_dim, layer_type)
        for encoding_dim in ENCODING_DIMS
        for layer_type in LAYERS_TYPES
    ]

    pretext_datamodule = build_datamodule()

    # train all backbones
    for encoding_dim, layer_type in combinations:
        pretext_model = build_pretext_model(encoding_dim, layer_type)
        pretext_trainer = build_lightning_trainer(f"TFC_{encoding_dim}_{layer_type}")
        pretext_trainer.fit(pretext_model, pretext_datamodule)

        plot_train_val(
            f"lightning_logs/TFC_{encoding_dim}_{layer_type}/version_0/",
            f"Training and validation loss\nconfig:{encoding_dim}_{layer_type}",
            f"figures/train_val_{encoding_dim}_{layer_type}.png",
        )



def fit_multiple_configs(ENCODING_DIMS):
    LAYERS_TYPES = ["conv1d", "mlp"]

    combinations = [
        (encoding_dim, layer_type)
        for encoding_dim in ENCODING_DIMS
        for layer_type in LAYERS_TYPES
    ]

    downstream_datamodule = build_datamodule("downstream")

    if os.path.exists("results/results_hyperparams.csv"):
        results = pd.read_csv("results/results_hyperparams.csv").to_dict("records")
    else:
        results = []

    
    # train downstreams with the three settings
    for encoding_dim, layer_type in combinations:

        # supervised learning
        pretext_model = build_pretext_model(encoding_dim, layer_type)
        downstream_model = tfc.TFCModel(pretext_model.backbone, learning_mode="finetune")
        downstream_trainer = build_lightning_trainer(
            f"TFC_downstream_supervised_{encoding_dim}_{layer_type}"
        )
        downstream_trainer.fit(downstream_model, downstream_datamodule)
        
        results = evaluate_model(
            downstream_model,
            downstream_datamodule,
            results,
            encoding_dim,
            layer_type,
            "supervised"
        )

        # finetune
        pretext_model = build_pretext_model(encoding_dim, layer_type)
        checkpoint = torch.load(
            f"lightning_logs/TFC_{encoding_dim}_{layer_type}/version_0/checkpoints/TFC-model.ckpt"
        )
        pretext_model.load_state_dict(checkpoint["state_dict"])
        downstream_model = tfc.TFCModel(pretext_model.backbone, learning_mode="finetune")
        downstream_trainer = build_lightning_trainer(
            f"TFC_downstream_finetune_{encoding_dim}_{layer_type}"
        )
        downstream_trainer.fit(downstream_model, downstream_datamodule)

        results = evaluate_model(
            downstream_model,
            downstream_datamodule,
            results,
            encoding_dim,
            layer_type,
            "finetune"
        )


        # freeze
        pretext_model = build_pretext_model(encoding_dim, layer_type)
        checkpoint = torch.load(
            f"lightning_logs/TFC_{encoding_dim}_{layer_type}/version_0/checkpoints/TFC-model.ckpt"
        )
        pretext_model.load_state_dict(checkpoint["state_dict"])
        downstream_model = tfc.TFCModel(pretext_model.backbone, learning_mode="freeze")
        downstream_trainer = build_lightning_trainer(
            f"TFC_downstream_freeze_{encoding_dim}_{layer_type}"
        )
        downstream_trainer.fit(downstream_model, downstream_datamodule)


        results = evaluate_model(
            downstream_model,
            downstream_datamodule,
            results,
            encoding_dim,
            layer_type,
            "freeze"
        )


        pd.DataFrame(results).to_csv("results/results_hyperparams.csv", index = False)


def fit_small_data():
    layer_type = "conv1d"
    encoding_dim = 128
    learning_mode = "freeze"
    results = []

    # fit pretext
    pretext_model = build_pretext_model(encoding_dim, layer_type)
    pretext_trainer = build_lightning_trainer(f"TFC_small_{encoding_dim}_{layer_type}")
    pretext_datamodule = build_datamodule()
    pretext_trainer.fit(pretext_model, pretext_datamodule)

    # make a copy of the backbone
    pretext_model_copy = build_pretext_model(encoding_dim, layer_type)
    pretext_model_copy.load_state_dict(pretext_model.state_dict())

    DATA_FRACTION = [i/10 for i in range(1, 11)]
    for frac in DATA_FRACTION:
        downstream_model = tfc.TFCModel(
            pretext_model.backbone, learning_mode=learning_mode
        )
        downstream_datamodule = build_datamodule("downstream", frac)
        downstream_trainer = build_lightning_trainer(
            f"TFC_downstream_small_{encoding_dim}_{layer_type}_{learning_mode}"
        )
        downstream_trainer.fit(downstream_model, downstream_datamodule)

        train_acc, validation_acc, test_acc = eval_linear_model(
            downstream_model, downstream_datamodule
        )
        train_acc_knn, validation_acc_knn, test_acc_knn = eval_knn(
            downstream_model, downstream_datamodule
        )

        results.append(
            {
                "encoding_dim": encoding_dim,
                "layer_type": layer_type,
                "learning_mode": learning_mode,
                "train_acc": train_acc,
                "validation_acc": validation_acc,
                "test_acc": test_acc,
                "train_acc_knn": train_acc_knn,
                "validation_acc_knn": validation_acc_knn,
                "test_acc_knn": test_acc_knn,
                "frac": frac,
            }
        )

        pd.DataFrame(results).to_csv("results/experiments_small_data.csv")
        pretext_model.load_state_dict(pretext_model_copy.state_dict())


if __name__ == "__main__":
    if not os.path.exists("figures"):
        os.makedirs("figures")

    if not os.path.exists("results"):
        os.makedirs("results")

    ENCODING_DIMS = [32, 64, 128, 256]
    #fit_small_data()
    #fit_backbones(ENCODING_DIMS)
    fit_multiple_configs(ENCODING_DIMS)
