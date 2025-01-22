from torch import nn
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor
import torch.nn.functional as F
import torch
import torch.distributed as dist
import lightning as L

class BT_ProjectionHead(nn.Module):
    def __init__(
        self,
        blocks: Sequence[
            Union[
                Tuple[int, int, Optional[nn.Module], Optional[nn.Module]],
                Tuple[int, int, Optional[nn.Module], Optional[nn.Module], bool],
            ],
        ]
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for block in blocks:
            input_dim, output_dim, batch_norm, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    # Default implementation
    def preprocess_step(self, x): return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocess_step(x)
        projection: Tensor = self.layers(x)
        return projection
    
class BarlowTwins(L.LightningModule):
    def __init__(self, 
                 backbone, 
                 projection_head,
                 loss_fn = None,
                 learning_rate = 0.06):
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = BarlowTwinsLoss()
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 0.06


    def forward(self, x):
        x = self.backbone(x)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x, x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.loss_fn(z0, z1)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optim

class BarlowTwinsLoss(nn.Module):
    def __init__(self, 
                 lambda_param: float = 5e-3, 
                 gather_distributed: bool = False):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        # normalize repr. along the batch dimension
        z_a_norm, z_b_norm = _normalize(z_a, z_b)

        N = z_a.size(0)

        # cross-correlation matrix
        c = z_a_norm.T @ z_b_norm
        c.div_(N)

        # sum cross-correlation matrix between multiple gpus
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                c = c / world_size
                dist.all_reduce(c)

        invariance_loss = torch.diagonal(c).add_(-1).pow_(2).sum()
        redundancy_reduction_loss = _off_diagonal(c).pow_(2).sum()
        loss = invariance_loss + self.lambda_param * redundancy_reduction_loss

        return loss

def _normalize(
    z_a: Tensor, z_b: Tensor
) -> Tuple[Tensor, Tensor]:
    """Helper function to normalize tensors along the batch dimension."""
    combined = torch.stack([z_a, z_b], dim=0)  # Shape: 2 x N x D
    normalized = F.batch_norm(
        combined.flatten(0, 1),
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=True,
    ).view_as(combined)
    return normalized[0], normalized[1]

def _off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()