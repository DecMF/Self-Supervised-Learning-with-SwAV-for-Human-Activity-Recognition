import lightning as L
import torch
from torchmetrics.functional import accuracy


class MultiLayerPerceptron(L.LightningModule):
    def __init__(
        self,
        input_features: int = 360,
        hidden_size: int = 64,
        num_classes: int = 6,
        learning_rate: float = 0.001,
    ):
        """Simple MultiLayer Perceptron model with ReLU activation function.

        Parameters
        ----------
        input_features : int, optional
            Number of input features, by default 360
        hidden_size : int, optional
            Number of neurons in the first hidden layer, by default 64
        num_classes : int, optional
            Number of classes, by default 6
        learning_rate : float, optional
            Learning rate for Adam optimizer, by default 0.001
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.block = torch.nn.Sequential(
            torch.nn.Linear(input_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes),
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        return self.block(x)

    def _common_step(
        self, batch: torch.Tensor, stage_name: str
    ) -> torch.Tensor:
        """Common step for training, validation and test steps.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of data. Each batch is a 2-element tuple (x, y) where x is
            the input data and y is the target label.
        stage_name : str
            Name of the stage (train, val, test), used for logging.

        Returns
        -------
        torch.Tensor
            Loss value for the given batch.
        """
        # Unpack
        x, y = batch
        
        # Forward pass
        logits = self.forward(x)
        # Calculate loss
        loss = self.loss(logits, y)
        # Log loss
        self.log(f"{stage_name}_loss", loss, on_epoch=True, prog_bar=True)

        # Calculate predictions and accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy_score = accuracy(
            predictions, y, num_classes=self.num_classes, task="multiclass"
        )
        # Log accuracy
        self.log(
            f"{stage_name}_acc", accuracy_score, on_epoch=True, prog_bar=True
        )

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        loss = self._common_step(batch, stage_name="train")
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        loss = self._common_step(batch, stage_name="val")
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        loss = self._common_step(batch, stage_name="test")
        return loss

    def configure_optimizers(self):
        # self.parameters() is a method from LightningModule that returns all
        # the parameters of this model (self.block in this case)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
