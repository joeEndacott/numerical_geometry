"""
Neural network
==============

This module provides a PyTorch-based neural network designed to learn deformation fields. It also
groups functions related to training neural networks.

TODO: remeshing via decimation to fix batch problem.
TODO: curriculum learning.
TODO: add function which finds the optimal learning rate.
"""

import time
import copy
from typing import Protocol
import torch
from torch import nn
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class LossFunctionProtocol(Protocol):
    """
    Loss function protocol
    ======================

    This class defines the protocol for loss functions used in neural network training.
    """

    def __call__(
        self,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        deformation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss between the deformed source point cloud and the target point cloud
        """
        ...  # pylint: disable=unnecessary-ellipsis


class NeuralNetwork(nn.Module):
    """
    Neural network
    ==============

    A class representing a simple neural network, designed to learn deformation fields. The class
    also groups functions relating to the neural network.
    """

    allowed_input_dim = [3, 6]
    allowed_output_dim = [3]

    def __init__(
        self,
        layers: int = 4,
        input_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 3,
    ):
        # Error checking.
        if input_dim not in NeuralNetwork.allowed_input_dim:
            raise ValueError(
                f"input_dim must be one of {NeuralNetwork.allowed_input_dim}."
            )
        if output_dim not in NeuralNetwork.allowed_output_dim:
            raise ValueError(
                f"output_dim must be one of {NeuralNetwork.allowed_output_dim}."
            )

        # Add input and output dimension as attributes.
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize the neural network.
        super().__init__()
        mods = []
        for _ in range(layers):
            mods += [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
            input_dim = hidden_dim
        mods += [nn.Linear(input_dim, output_dim)]
        self.net = nn.Sequential(*mods)

        # Add number of parameters as an attribute.
        self.num_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x):  # pylint: disable=missing-function-docstring
        return self.net(x)

    def _sync(self, device: torch.device):
        """
        Sync
        ====

        Synchronize device operations for accurate timing.
        """
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    def find_optimal_lr(
        self,
        source_mesh: pv.PolyData,
        target_mesh: pv.PolyData,
        device: torch.device,
        loss_function: LossFunctionProtocol,
        *,
        # Core parameters.
        optimizer_type: str = "SGD",
        initial_lr: float = 1e-4,
        final_lr: float = 1,
        num_iterations: int = 1000,
        # Batch parameters.
        batch_size: int = 1024,
        # Smoothing parameters.
        window_length: int = 51,
        polyorder: int = 5,
        # Early stopping parameters.
        early_stopping: bool = False,
        loss_threshold: float = 1,
        min_iterations: int = 20,
        # Debugging and monitoring.
        verbose: bool = True,
    ):
        """
        Find optimal learning rate
        ==========================

        Find the optimal learning rate using a learning rate range test.
        """
        if verbose:
            print("Configuration:")
            print("=" * len("Configuration:"))
            print(f"Optimizer:          {optimizer_type}")
            print(f"Learning rate:      [{initial_lr}, {final_lr}]")
            print(f"Early stopping:     {early_stopping}")
        if early_stopping:
            print(f"    Loss threshold: {loss_threshold}")
        print()

        # Save the initial model state.
        initial_state = copy.deepcopy(self.state_dict())
        self.train()

        try:
            # Prepare the data.
            source, target = PreparationUtils.prepare_data(
                self, source_mesh, target_mesh, device
            )

            # Initialize an optimizer.
            optimizer = TrainingUtils.get_optimizer(self, optimizer_type, initial_lr)

            # Generate log-spaced learning rates.
            log_learning_rates = np.linspace(
                np.log(initial_lr), np.log(final_lr), num_iterations
            )
            learning_rates = np.exp(log_learning_rates)

            losses = []
            for iteration in range(num_iterations):
                # Set the learning rate for this iteration.
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rates[iteration]

                # Generate a source batch.
                batch_size = min(batch_size, source.size(0))
                indices = torch.randperm(source.size(0))[:batch_size]
                batch_source = source[indices]

                # Generate a target batch.
                target_batch_size = min(batch_size, target.size(0))
                target_indices = torch.randperm(target.size(0))[:target_batch_size]
                batch_target = target[target_indices]

                # Compute the deformation and loss for this batch.
                optimizer.zero_grad()
                batch_deformation = self(batch_source)
                batch_loss = loss_function(
                    batch_source[:, :3], batch_target, batch_deformation
                )
                current_loss = batch_loss.item()
                losses.append(current_loss)

                # Check early stopping.
                if early_stopping:
                    if current_loss > loss_threshold and iteration >= min_iterations:
                        print(
                            f"\n\nEarly stopping triggered at lr = {learning_rates[iteration]}."
                        )
                        break

                # Backpropagate the loss.
                batch_loss.backward()
                optimizer.step()

                # Print the progress.
                DisplayUtils.print_progress_bar(
                    iteration + 1,
                    num_iterations,
                    current_loss,
                    learning_rates[iteration],
                    40,
                )

        finally:
            # Restore original self state
            self.load_state_dict(initial_state)
            self.eval()
            self.zero_grad()

        # Smooth the losses.
        smoothed_losses = savgol_filter(
            np.array(losses), window_length=window_length, polyorder=polyorder
        )

        # Ensure all arrays are the same length
        min_length = min(len(learning_rates), len(losses), len(smoothed_losses))
        learning_rates = learning_rates[:min_length]
        losses = losses[:min_length]
        smoothed_losses = smoothed_losses[:min_length]

        if verbose:
            if early_stopping:
                print("\nResults:")
            else:
                print("\n\nResults:")
            print("=" * len("Results:"))

        # Plot loss vs. learning rate.
        DisplayUtils.plot_optimal_lr(
            learning_rates,
            losses,
            smoothed_losses,
        )

    def train_model(
        self,
        source_mesh: pv.PolyData,
        target_mesh: pv.PolyData,
        device: torch.device,
        loss_function: LossFunctionProtocol,
        *,
        # Core training parameters.
        epochs: int = 100,
        optimizer_type: str = "SGD",
        lr: float = 1e-3,
        # Batch parameters.
        source_batch_size: int = 2048,
        target_batch_size: int = 2048,
        # Early stopping parameters.
        early_stopping: bool = False,
        patience: int = 20,
        min_delta: float = 1e-6,
        min_epochs: int = 20,
        # Debugging and monitoring.
        verbose: bool = True,
        plot_results: bool = True,
    ):
        """
        Train model
        ===========

        Trains the model to learn a deformation field from the source to the target.
        """
        source_batch_size = min(source_batch_size, source_mesh.points.shape[0])
        target_batch_size = min(target_batch_size, target_mesh.points.shape[0])

        if verbose:
            print("Configuration:")
            print("=" * len("Configuration:"))
            print(f"Source points:      {source_mesh.points.shape[0]}")
            print(f"Target points:      {target_mesh.points.shape[0]}")
            print(f"Epochs:             {epochs}")
            print(f"Optimizer:          {optimizer_type}")
            print(f"Learning rate:      {lr}")
            print(f"Source batch size:  {source_batch_size}")
            print(f"Target batch size:  {target_batch_size}")
            print(f"Learning rate:      {lr}")
            print(f"Early stopping:     {early_stopping}")
        if early_stopping:
            print(f"    Patience:       {patience}")
            print(f"    Min delta:      {min_delta}")
        print()

        prep_start = time.time()
        self.train()

        try:
            # Extract the source and target.
            source, target = PreparationUtils.prepare_data(
                self, source_mesh, target_mesh, device
            )

            # Initialize the optimizer and scheduler.
            optimizer = TrainingUtils.get_optimizer(self, optimizer_type, lr)

            # Compute the preparation time.
            prep_time = time.time() - prep_start

            # Initialize variables to store training performance.
            epoch_history, loss_history, lr_history = [], [], []
            overhead_time, forward_time, loss_time, backward_time = (0, 0, 0, 0)
            stopping_triggered = False

            # Training loop.
            training_start = time.time()
            for epoch in range(epochs):
                epoch_loss, num_batches = 0.0, 0

                # Extract source and target batches for the current epoch.
                overhead_start = time.time()
                source_batches, target_batches = PreparationUtils.compute_epoch_batches(
                    source, target, source_batch_size, target_batch_size, device
                )
                max_num_batches = max(len(source_batches), len(target_batches))

                # Batches loop.
                for batch_idx in range(max_num_batches):
                    # Sample a subset of the source and target.
                    batch_source = source_batches[batch_idx % len(source_batches)]
                    batch_target = target_batches[batch_idx % len(target_batches)]
                    self._sync(device)
                    overhead_time_current = time.time() - overhead_start

                    # Compute the deformation for this batch.
                    forward_start = time.time()
                    optimizer.zero_grad()
                    batch_deformation = self(batch_source)
                    self._sync(device)
                    forward_time_current = time.time() - forward_start

                    # Compute the loss for this batch.
                    loss_start = time.time()
                    batch_loss = loss_function(
                        batch_source[:, :3], batch_target, batch_deformation
                    )
                    self._sync(device)
                    loss_time_current = time.time() - loss_start

                    # Backpropagate the loss.
                    backward_start = time.time()
                    batch_loss.backward()
                    optimizer.step()
                    self._sync(device)
                    backward_time_current = time.time() - backward_start

                    # Accumulate the times.
                    overhead_time += overhead_time_current
                    forward_time += forward_time_current
                    loss_time += loss_time_current
                    backward_time += backward_time_current

                    # Update the epoch loss and number of batches.
                    epoch_loss += batch_loss.item()
                    num_batches += 1
                    overhead_start = time.time()

                # Compute the loss and learning rate for this epoch.
                epoch_loss = epoch_loss / num_batches
                current_lr = optimizer.param_groups[0]["lr"]

                # Store values for plotting.
                epoch_history.append(epoch)
                loss_history.append(epoch_loss)
                lr_history.append(current_lr)

                # Check early stopping.
                if early_stopping:
                    if TrainingUtils.should_stop_early(
                        loss_history, patience, min_delta, min_epochs
                    ):
                        stopping_triggered = True
                        print(f"\n\nEarly stopping triggered at epoch {epoch+1}.")
                        break

                # Update the progress bar.
                DisplayUtils.print_progress_bar(
                    epoch + 1,
                    epochs,
                    epoch_loss,
                    current_lr,
                    bar_length=40,
                )

            training_time = time.time() - training_start

            # Print the results.
            if verbose:
                total_time = prep_time + training_time
                accounted_time = (
                    overhead_time + forward_time + loss_time + backward_time
                )
                unaccounted_time = training_time - accounted_time

                if stopping_triggered:
                    print("\nResults:")
                else:
                    print("\n\nResults:")
                print("=" * len("Results:"))
                print(f"Training completed after {len(loss_history)} epochs.")
                print(f"Total time:      {prep_time+training_time:.2f}s")
                print(f"    Preparation: {prep_time/total_time:.1%}.")
                print(f"    Overhead:    {overhead_time/total_time:.1%}.")
                print(f"    Forward:     {forward_time/total_time:.1%}.")
                print(f"    Loss:        {loss_time/total_time:.1%}.")
                print(f"    Backward:    {backward_time/total_time:.1%}.")
                print(f"    Unaccounted: {unaccounted_time/total_time:.1%}.")

            # Plot the results.
            if plot_results:
                DisplayUtils.plot_loss_and_lr(epoch_history, loss_history)

            # Return training results.

        finally:
            self.eval()

    def evaluate_model(
        self, source_mesh: pv.PolyData, device: torch.device
    ) -> np.ndarray:
        """
        Evaluate model
        ==============

        Predicts the deformation field that maps the source mesh to the target mesh specified
        during training.
        """
        self.eval()

        with torch.no_grad():
            # Calculate the point cloud.
            source_points = torch.from_numpy(source_mesh.points).float().to(device)

            # Prepare the source, and calculate the deformation.
            if self.input_dim == 3:
                source = source_points
                return self(source).to("cpu").detach().numpy()
            if self.input_dim == 6:
                source_mesh = source_mesh.compute_normals()
                source_normals = (
                    torch.from_numpy(source_mesh.point_data["Normals"])
                    .float()
                    .to(device)
                )
                source = torch.cat((source_points, source_normals), dim=1)
                return self(source).to("cpu").detach().numpy()
            raise ValueError("self.input_dim must be 3 or 6.")


class LossFunction:
    """
    Loss function
    =============

    A class to group functions related to loss functions.
    """

    @staticmethod
    def chamfer_distance(
        source_points: torch.Tensor, target_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Chamfer distance
        ================

        Computes the Chamfer distance between two point clouds, source_points and target_points.
        """
        # Compute pairwise squared distances between points in source_points and points in
        # target_points.
        source_points_expanded = source_points.unsqueeze(1)
        target_points_expanded = target_points.unsqueeze(0)
        distances = torch.sum(
            (source_points_expanded - target_points_expanded) ** 2, dim=2
        )

        # For each point in source_points, find nearest neighbor in target_points.
        source_min_distance, _ = torch.min(distances, dim=1)

        # For each point in target_points, find nearest neighbor in source_points.
        target_min_distance, _ = torch.min(distances, dim=0)

        # Take the average of the mean nearest neighbor distances.
        return torch.mean(source_min_distance) + torch.mean(target_min_distance)

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
        source_points: torch.Tensor,
        deformation: torch.Tensor,
        num_neighbors: int = 8,
    ) -> torch.Tensor:
        """
        Laplacian loss
        ==============

        Laplacian regularization for a point cloud deformation. The function returns the average
        magnitude of the Laplacian of the deformation field.
        """
        # Compute pairwise distances.
        pairwise_distances = torch.cdist(source_points, source_points)

        # Find the mean nearest neighbor deformations.
        neighbor_indices = pairwise_distances.topk(
            num_neighbors + 1, largest=False
        ).indices[:, 1:]
        mean_nn_deformation = deformation[neighbor_indices].mean(dim=1)

        # Find the distance between each deformation and the mean nn deformation.
        laplacian = deformation - mean_nn_deformation

        # Calculate the Laplacian loss.
        return torch.sqrt((laplacian**2).mean())

    # @staticmethod
    # def edge_length_loss(
    #     source_points: torch.Tensor, deformation: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Edge length loss
    #     ================

    #     Edge length regularization. The function returns the average change in the edge lengths.
    #     """
    #     # Compute the deformed source.
    #     deformed_source = source_points + deformation

    #     # Find the original and deformed edge lengths.
    #     original_lengths = torch.norm(
    #         source_points[edges[:, 0]] - source_points[edges[:, 1]], dim=1
    #     )
    #     deformed_lengths = torch.norm(
    #         deformed_source[edges[:, 0]] - deformed_source[edges[:, 1]], dim=1
    #     )

    #     # Calculate the average squared difference.
    #     return ((deformed_lengths - original_lengths) ** 2).mean()


class PreparationUtils:
    """
    Preparation utils
    =================

    A class to group utility functions related to preparing data for training.
    """

    @staticmethod
    def prepare_data(
        model: NeuralNetwork,
        source_mesh: pv.PolyData,
        target_mesh: pv.PolyData,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data
        ============

        Extracts data relevant for training. Returns source and target point clouds and source
        vertex normals (optional).
        """
        if source_mesh.points is None or len(source_mesh.points) == 0:
            raise ValueError("source_mesh cannot be empty.")
        if target_mesh.points is None or len(target_mesh.points) == 0:
            raise ValueError("target_mesh cannot be empty.")

        if model.input_dim == 3:
            source = torch.from_numpy(source_mesh.points).float().to(device)
            target = torch.from_numpy(target_mesh.points).float().to(device)
            return source, target

        if model.input_dim == 6:
            # Compute normals if they don't exist
            if "Normals" not in source_mesh.point_data:
                source_mesh = source_mesh.compute_normals()
            if "Normals" not in source_mesh.point_data:
                raise ValueError("Failed to compute normals for source_mesh.")

            # Extract source points and normals, and concatenate them.
            source_points = torch.from_numpy(source_mesh.points).float().to(device)
            source_normals = (
                torch.from_numpy(source_mesh.point_data["Normals"]).float().to(device)
            )
            source = torch.cat((source_points, source_normals), dim=1)
            target = torch.from_numpy(target_mesh.points).float().to(device)
            return source, target

        raise ValueError("model.input_dim must be equal to 3 or 6.")

    @staticmethod
    def precompute_all_batches(
        source: torch.Tensor,
        target: torch.Tensor,
        source_batch_size: int,
        target_batch_size: int,
        device: torch.device,
        epochs: int,
    ) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        """
        Precompute all batches
        ======================

        Precompute source and target batches for all epochs.
        """
        # Error checking for invalid batch sizes.
        if source_batch_size <= 0:
            raise ValueError("source_batch_size must be greater than 0.")
        if target_batch_size <= 0:
            raise ValueError("target_batch_size must be greater than 0.")
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0.")
        if source.shape[0] == 0:
            raise ValueError("source tensor cannot be empty.")
        if target.shape[0] == 0:
            raise ValueError("target tensor cannot be empty.")

        # Validate source_batch_size and target_batch_size.
        source_batch_size = min(source_batch_size, source.shape[0])
        target_batch_size = min(target_batch_size, target.shape[0])

        # Initialize source batches.
        all_source_batches = []
        num_source_batches = (
            source.size(0) + source_batch_size - 1
        ) // source_batch_size

        # Initialize target batches.
        all_target_batches = []
        num_target_batches = (
            target.size(0) + target_batch_size - 1
        ) // target_batch_size

        # Pre-compute batches for all epochs.
        for _ in range(epochs):
            # Shuffle the source and target indices.
            source_indices = torch.randperm(source.size(0), device=device)
            target_indices = torch.randperm(target.size(0), device=device)

            # Create source batches.
            epoch_source_batches = []
            for batch_idx in range(num_source_batches):
                start_idx = batch_idx * source_batch_size
                end_idx = min(start_idx + source_batch_size, source.size(0))
                batch_indices = source_indices[start_idx:end_idx]
                epoch_source_batches.append(source[batch_indices])

            # Create target batches.
            epoch_target_batches = []
            for batch_idx in range(num_target_batches):
                start_idx = batch_idx * target_batch_size
                end_idx = min(start_idx + target_batch_size, target.size(0))
                batch_indices = target_indices[start_idx:end_idx]
                epoch_target_batches.append(target[batch_indices])

            all_source_batches.append(epoch_source_batches)
            all_target_batches.append(epoch_target_batches)

        return all_source_batches, all_target_batches

    @staticmethod
    def compute_epoch_batches(
        source: torch.Tensor,
        target: torch.Tensor,
        source_batch_size: int,
        target_batch_size: int,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Generate epoch batches
        ======================

        Generate source and target batches for a single epoch.
        """
        # Shuffle indices for this epoch.
        source_indices = torch.randperm(source.size(0), device=device)
        target_indices = torch.randperm(target.size(0), device=device)

        # Create source batches
        source_batches = []
        for start_idx in range(0, source.size(0), source_batch_size):
            end_idx = min(start_idx + source_batch_size, source.size(0))
            batch_indices = source_indices[start_idx:end_idx]
            source_batches.append(source[batch_indices])

        # Create target batches
        target_batches = []
        for source_idx in range(0, target.size(0), source_batch_size):
            end_idx = min(source_idx + target_batch_size, target.size(0))
            batch_indices = target_indices[source_idx:end_idx]
            target_batches.append(target[batch_indices])

        return source_batches, target_batches


class TrainingUtils:
    """
    Training utils
    ==============

    A class to group utility functions related to training neural networks.
    """

    optimizers = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "RMSprop": torch.optim.RMSprop,
        "Adagrad": torch.optim.Adagrad,
        "Adadelta": torch.optim.Adadelta,
        "Adamax": torch.optim.Adamax,
        "NAdam": torch.optim.NAdam,
    }

    @staticmethod
    def get_optimizer(
        model: NeuralNetwork, optimizer_type: str, lr: float
    ) -> torch.optim.Optimizer:
        """
        Get optimizer
        =============

        Initializes and returns the optimizer.
        """

        if optimizer_type not in TrainingUtils.optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_type}.")
        return TrainingUtils.optimizers[optimizer_type](model.parameters(), lr=lr)

    @staticmethod
    def get_scheduler(
        scheduler_type: str | None,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        epochs: int,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """
        Get scheduler
        =============

        Initializes and returns the scheduler. If scheduler_type is None, None is returned.
        """
        if scheduler_type is None:
            return None
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=initial_lr * 0.1
            )
        if scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        if scheduler_type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=50
            )
        raise ValueError(f"Unknown scheduler type: {scheduler_type}.")

    @staticmethod
    def should_stop_early(
        loss_history: list[float], patience: int, min_delta: float, min_epochs: int
    ):
        """
        Should stop early
        =================

        Checks if training should stop early.
        """
        if len(loss_history) < min_epochs:
            return False
        if len(loss_history) < patience + 1:
            return False

        # Compare the best losses from the previous 2 epoch windows.
        best_recent_loss = min(loss_history[-patience:])
        best_old_loss = min(loss_history[-2 * patience : -patience])
        return (best_old_loss - best_recent_loss) < min_delta


class DisplayUtils:
    """
    Display utils
    =============

    A class to group functions related to displaying training results.
    """

    @staticmethod
    def print_progress_bar(
        epoch: int,
        epochs: int,
        loss: float,
        lr: float,
        bar_length: int,
    ):
        """
        Print progress bar
        ==================

        Displays a progress bar.
        """
        filled_length = int(bar_length * epoch / epochs)
        progress_bar = "█" * filled_length + "-" * (bar_length - filled_length)

        print(
            f"\r[{progress_bar}] {epoch}/{epochs} ({(epoch/epochs):.1%}) | "
            f"Loss: {loss:.6f} | "
            f"Learning rate: {lr:.4f}",
            end="",
            flush=True,
        )

    @staticmethod
    def plot_loss_and_lr(
        epoch_history: list[int],
        loss_history: list[float],
    ):
        """
        Plot loss and learning rate
        ===========================

        Plots the loss and learning rate upon completion of training.
        """
        # Initialize the plot.
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot the loss.
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax1.scatter(epoch_history, loss_history, s=10)

        # Plot the learning rate.
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.scatter(epoch_history, loss_history, s=10)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_optimal_lr(
        learning_rates: np.ndarray,
        losses: np.ndarray,
        smoothed_losses: np.ndarray,
    ):
        """
        Plot optimal learning rate
        ==========================

        Plot the learning rate test results.
        """
        _, ax = plt.subplots(figsize=(10, 8))

        # Plot loss vs learning rate on a log scale.
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", axis="both", alpha=0.3)
        ax.semilogx(learning_rates, losses, alpha=0.6, label="Raw Loss")
        ax.semilogx(learning_rates, smoothed_losses, linewidth=2, label="Smoothed Loss")
        ax.legend()
        plt.tight_layout()
        plt.show()
