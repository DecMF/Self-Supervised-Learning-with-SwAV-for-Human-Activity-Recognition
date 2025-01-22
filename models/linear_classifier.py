import lightning as L
import torch


class LinearClassifier(L.LightningModule):
    def __init__(
        self,
        backbone: L.LightningModule,
        head: L.LightningModule,
        num_classes: int = 6,
        learning_rate: float = 0.001,
        flatten: bool = True,
        freeze_backbone: bool = False,
        loss_fn: torch.nn.modules.loss._Loss = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.flatten = flatten
        self.loss_fn = loss_fn
        self.freeze_backbone = freeze_backbone

        if self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def calculate_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, stage_name: str
    ) -> dict:
        """Calculate metrics for the given batch.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted labels.
        y_true : torch.Tensor
            True labels.

        Returns
        -------
        dict
            Dictionary of metrics.
        """
        assert stage_name in [
            "train",
            "val",
            "test",
        ], f"Invalid stage name: {stage_name}"

        # Our metrics dictionary
        metrics = dict()

        # Move to CPU and detach
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()

        # Calculate accuracy
        y_pred = torch.argmax(y_pred, dim=1)
        acc = float((y_pred == y_true).float().mean())
        metrics = {f"{stage_name}_accuracy": acc}

        # Add more metrics if wanted...., e.g. f1, precision, recall, etc.
        # ...

        return metrics

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        if self.flatten:
            x = x.view(x.size(0), -1)
        return self.head(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # Unpack
        x, y = batch
        # Forward pass
        logits = self.forward(x)
        # Calculate loss
        loss = self.loss_fn(logits, y)
        # Log loss
        self.log(
            f"train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )

        # return a dictionary of metrics (loss must be present)
        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # Unpack
        x, y = batch
        # Forward pass
        logits = self.forward(x)
        # Calculate loss
        loss = self.loss_fn(logits, y)
        # Log loss
        self.log(f"val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # calculate metrics and get a dictionary of metrics and log all metrics
        metrics = self.calculate_metrics(logits, y, stage_name="val")
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        # return a dictionary of metrics (loss must be present)
        metrics["loss"] = loss
        return metrics

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        # Unpack
        x, y = batch
        # Forward pass
        logits = self.forward(x)
        # Calculate loss
        loss = self.loss_fn(logits, y)
        # Log loss
        self.log(f"test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # calculate metrics and get a dictionary of metrics and log all metrics
        metrics = self.calculate_metrics(logits, y, stage_name="test")
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        # return a dictionary of metrics (loss must be present)
        metrics["loss"] = loss
        metrics["y_true"] = y
        metrics["y_pred"] = logits
        return metrics

    def _freeze(self, model):
        """Freezes the model, i.e. sets the requires_grad parameter of all the
        parameters to False.

        Parameters
        ----------
        model : _type_
            The model to freeze
        """
        for param in model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        """Configures the optimizer. If ``update_backbone`` is True, it will
        update the parameters of the backbone and the head. Otherwise, it will
        only update the parameters of the head.
        """
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)