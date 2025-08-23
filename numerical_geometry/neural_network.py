"""
Neural network
==============

This module provides a PyTorch-based neural network designed to learn deformation fields. It also
groups functions related to training neural networks.
"""

from typing import Union
import pyvista as pv
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):
    """
    Neural network
    ==============

    A class representing a simple neural network, designed to learn deformation fields. The class
    also groups functions relating to the neural network.
    """

    def __init__(self, parameters=128, layers=4, input_dim=3, output_dim=3):
        super().__init__()
        mods = []
        for _ in range(layers):
            mods += [nn.Linear(input_dim, parameters), nn.ReLU(inplace=True)]
            input_dim = parameters
        mods += [nn.Linear(input_dim, output_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        """
        Forward
        =======
        """
        return self.net(x)

    def train_model(
        self,
        source_mesh: Union[pv.PolyData, torch.Tensor],
        target_mesh: Union[pv.PolyData, torch.Tensor],
        device: torch.device,
        loss_function,
        optimizer_type: str = "SGD",
        epochs: int = 100,
        batch_size: int = 512,
        target_batch_size: int = 1000,
        learning_rate: float = 1e-3,
        print_every: int = 10,
        plot_loss: bool = True,
        return_history: bool = False,
    ):
        """
        Train model
        ===========

        Trains the model to learn a deformation field from the source to the target.
        """
        # Prepare point clouds.
        if isinstance(source_mesh, pv.PolyData):
            source = torch.from_numpy(source_mesh.points).float().to(device)
        elif isinstance(source_mesh, torch.Tensor):
            source = source_mesh.to(device)
        else:
            raise ValueError("source_mesh must be a PyVista mesh or a PyTorch tensor.")

        if isinstance(target_mesh, pv.PolyData):
            target = torch.from_numpy(target_mesh.points).float().to(device)
        elif isinstance(target_mesh, torch.Tensor):
            target = target_mesh.to(device)
        else:
            raise ValueError("target_mesh must be a PyVista mesh or a PyTorch tensor.")

        # Valid batch_size and target_batch_size.
        if batch_size > source.shape[0]:
            batch_size = source.shape[0]

        if target_batch_size > target.shape[0]:
            target_batch_size = target.shape[0]

        # Create a dataset and dataloader for source.
        source_dataset = TensorDataset(source)
        source_dataloader = DataLoader(
            source_dataset, batch_size=batch_size, shuffle=True
        )

        # Initialize the optimizer.
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError("Unknown optimizer type.")

        loss_history = []
        epoch_history = []

        # Training loop.
        for epoch in torch.arange(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for _, (batch_source,) in enumerate(source_dataloader):
                # Sample a subset of the target.
                target_indices = torch.randperm(target.size(0))[:target_batch_size]
                batch_target = target[target_indices]

                # Compute the deformation for this batch.
                batch_deformation = self(batch_source)

                # Compute the loss for this batch.
                batch_loss = loss_function(
                    batch_source, batch_target, batch_deformation
                )

                # Backpropagate the loss.
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += batch_loss.item()
                num_batches += 1

            # Average loss for this epoch.
            average_epoch_loss = epoch_loss / num_batches

            # Store loss values for plotting.
            loss_history.append(average_epoch_loss)
            epoch_history.append(epoch)

            if epoch % print_every == 0:
                print(f"[{epoch}/{epochs}]: loss = {average_epoch_loss:.6f}")

        if plot_loss:
            _, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.scatter(epoch_history, loss_history, s=10)
            plt.show()
        if return_history:
            return loss_history, epoch_history

    def get_deformation_field(
        self, source_mesh: Union[pv.PolyData, torch.Tensor], device: torch.device
    ):
        """
        Get deformation field
        =====================

        Predicts the deformation field that maps the source mesh to the target mesh specified
        during training.
        """
        if isinstance(source_mesh, pv.PolyData):
            source = torch.from_numpy(source_mesh.points).float().to(device)
            return self(source).to("cpu").detach().numpy()
        elif isinstance(source_mesh, torch.Tensor):
            return self(source_mesh.to(device)).to("cpu").detach().numpy()
        else:
            raise ValueError("source_mesh must be a PyVista mesh or a PyTorch tensor.")


class LossFunction:
    """
    Loss function
    =============

    A class to group functions related to loss functions.
    """

    @staticmethod
    def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Chamfer distance
        ================

        Computes the Chamfer distance between two point clouds, x and y.
        """
        # Compute pairwise squared distances between points in x and points in y.
        x = x.unsqueeze(1)  # (N, 1, D)
        y = y.unsqueeze(0)  # (1, M, D)
        dist = torch.sum((x - y) ** 2, dim=2)  # (N, M)

        # For each point in x, find nearest neighbor in y.
        min_dist_x, _ = torch.min(dist, dim=1)

        # For each point in y, find nearest neighbor in x.
        min_dist_y, _ = torch.min(dist, dim=0)

        # Take the average of the mean nearest neighbor distances.
        return torch.mean(min_dist_x) + torch.mean(min_dist_y)

    @staticmethod
    def deformation_loss(deformation: torch.Tensor) -> torch.Tensor:
        """
        Deformation loss
        ================

        Deformation magnitude regularization for a point cloud deformation. The function returns
        the average size of the deformation field.
        """
        # Calculate the average deformation magnitude.
        return torch.sqrt((deformation**2).mean())

    @staticmethod
    def laplacian_loss(
        source: torch.Tensor, deformation: torch.Tensor, num_neighbors: int = 8
    ) -> torch.Tensor:
        """
        Laplacian loss
        ==============

        Laplacian regularization for a point cloud deformation. The function returns the average
        magnitude of the Laplacian of the deformation field.
        """
        # Compute pairwise distances.
        pairwise_distances = torch.cdist(source, source)

        # Find the mean nearest neighbor deformations.
        nn_idx = pairwise_distances.topk(num_neighbors + 1, largest=False).indices[
            :, 1:
        ]
        mean_nn_deformation = deformation[nn_idx].mean(dim=1)

        # Find the distance between each deformation and the mean nn deformation.
        laplacian = deformation - mean_nn_deformation

        # Calculate the Laplacian loss.
        return torch.sqrt((laplacian**2).mean())

    @staticmethod
    def edge_length_loss(
        source: torch.Tensor, deformation: torch.Tensor, edges: torch.Tensor
    ):
        """
        Edge length loss
        ================

        Edge length regularization. The function returns the average change in the edge lengths.
        """
        # Compute the deformed source.
        deformed_source = source + deformation

        # Find the original and deformed edge lengths.
        original_lengths = torch.norm(source[edges[:, 0]] - source[edges[:, 1]], dim=1)
        deformed_lengths = torch.norm(
            deformed_source[edges[:, 0]] - deformed_source[edges[:, 1]], dim=1
        )

        # Calculate the average squared difference.
        return ((deformed_lengths - original_lengths) ** 2).mean()
