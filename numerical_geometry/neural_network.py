# pylint: disable=too-many-lines
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
import gc
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

    This class represents a multilayer perceptron (MLP), designed to learn deformation fields. The
    class also groups functions relating to the neural network.
    """

    allowed_input_dim = [3, 6]
    allowed_output_dim = [3]

    def __init__(
        self,
        input_dim: int = 3,
        layers: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 3,
        dropout_prob: float = 0.1,
        *,
        verbose: bool = False,
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

        # Add inputs as attributes.
        self.input_dim = input_dim
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        # Initialize the neural network.
        super().__init__()
        mods = []
        for _ in range(layers):
            mods += [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob),
            ]
            input_dim = hidden_dim
        mods += [nn.Linear(input_dim, output_dim)]
        self.net = nn.Sequential(*mods)

        # Add number of parameters as an attribute.
        self.num_parameters = sum(p.numel() for p in self.parameters())

        # Print the configuration.
        if verbose:
            self.print_configuration()

    def forward(self, x):  # pylint: disable=missing-function-docstring
        return self.net(x)

    def print_configuration(self):
        """
        Print configuration
        ===================

        Prints the configuration of the model to the terminal.
        """
        print("Configuration:")
        print("=" * len("Configuration:"))
        print(f"Input dim:       {self.input_dim}")
        print(f"Layers:          {self.layers}")
        print(f"Hidden dim:      {self.hidden_dim}")
        print(f"Output dim:      {self.output_dim}")
        print(f"Num. parameters: {self.num_parameters:.3e}")
        print(f"Dropout prob.:   {self.dropout_prob}")

    def find_optimal_lr(
        self,
        source: pv.PolyData | torch.Tensor,
        target: pv.PolyData | torch.Tensor,
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
        plot_results: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find optimal learning rate
        ==========================

        Find the optimal learning rate using a learning rate range test.
        """
        # Extract the source and target.
        source = PreparationUtils.prepare_source(self, source, device)
        target = PreparationUtils.prepare_target(self, target, device)

        # Error checking.
        if num_iterations <= 0:
            raise ValueError("num_iterations must be greater than 0.")
        if initial_lr <= 0:
            raise ValueError("initial_lr must be greater than 0.")
        if final_lr <= 0:
            raise ValueError("final_lr must be greater than 0.")
        if final_lr <= initial_lr:
            raise ValueError("final_lr must be larger than initial_lr.")
        batch_size = min(batch_size, source.shape[0], target.shape[0])
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if window_length <= 0:
            raise ValueError("window_length must be greater than 0.")
        if polyorder <= 0:
            raise ValueError("polyorder must be greater than 0.")
        if min_iterations <= 0:
            raise ValueError("min_iterations must be greater than 0.")

        # Initialize an optimizer.
        optimizer = self._init_optimizer(optimizer_type, initial_lr)

        # Generate log-spaced learning rates.
        learning_rates = np.exp(
            np.linspace(np.log(initial_lr), np.log(final_lr), num_iterations)
        )

        # Initialize variables to store training performance.
        losses = []

        # Save the initial model state.
        initial_state = copy.deepcopy(self.state_dict())

        # Print the configuration.
        if verbose:
            print("Configuration:")
            print("=" * len("Configuration:"))
            print(f"Optimizer:          {optimizer_type}")
            print(f"Learning rate:      [{initial_lr}, {final_lr}]")
            print(f"Early stopping:     {early_stopping}")
            if early_stopping:
                print(f"    Loss threshold: {loss_threshold}")
        print()

        self.train()
        try:
            for iteration in range(num_iterations):
                loss_current = self._lr_test_loop(
                    source,
                    target,
                    device,
                    loss_function,
                    optimizer,
                    learning_rates[iteration],
                    batch_size,
                )
                losses.append(loss_current)

                # Check early stopping.
                if early_stopping:
                    if loss_current > loss_threshold and iteration >= min_iterations:
                        print(
                            f"\n\nEarly stopping triggered at lr = {learning_rates[iteration]}."
                        )
                        break

                # Print the progress.
                DisplayUtils.print_progress_bar(
                    iteration + 1,
                    num_iterations,
                    loss_current,
                    None,
                    learning_rates[iteration],
                    bar_length=40,
                )

        finally:
            # Restore the original model state.
            self.load_state_dict(initial_state)
            self.zero_grad()
            self.eval()
            NeuralNetwork._clear_cache(device)
            gc.collect()

        # Ensure learning_rates and losses are the same length
        min_length = min(len(learning_rates), len(losses))
        learning_rates = np.array(learning_rates[:min_length])
        losses = np.array(losses[:min_length])

        # Smooth the losses.
        smoothed_losses = savgol_filter(
            losses, window_length=window_length, polyorder=polyorder
        )

        if verbose:
            if early_stopping:
                print("\nResults:")
            else:
                print("\n\nResults:")
            print("=" * len("Results:"))

        # Plot loss vs. learning rate.
        if plot_results:
            DisplayUtils.plot_optimal_lr(
                learning_rates,
                losses,
                smoothed_losses,
            )

        return learning_rates, losses, smoothed_losses

    def train_model(
        self,
        source: pv.PolyData | torch.Tensor,
        target: pv.PolyData | torch.Tensor,
        device: torch.device,
        loss_function: LossFunctionProtocol,
        *,
        # Core training parameters.
        epochs: int = 100,
        optimizer_type: str = "SGD",
        lr: float = 1e-3,
        # Batch parameters.
        batch_size: int = 1024,
        # Early stopping parameters.
        early_stopping: bool = False,
        validation_fraction: float = 0.01,
        patience: int = 20,
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
        prep_start = time.time()

        # Extract the source and target.
        source = PreparationUtils.prepare_source(self, source, device)
        target = PreparationUtils.prepare_target(self, target, device)

        # Error checking.
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0.")
        if lr <= 0:
            raise ValueError("lr must be greater than 0.")
        batch_size = min(batch_size, source.shape[0], target.shape[0])
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if validation_fraction >= 1:
            raise ValueError("validation_fraction must be less than 1.")
        if patience <= 0:
            raise ValueError("patience must be greater than 0.")
        if min_epochs < 0:
            raise ValueError("min_epochs must be greater than or equal to 0.")

        # Prepare training and validation set.
        if early_stopping:
            training_source, validation_source = PreparationUtils.split_source(
                source, validation_fraction
            )
        else:
            training_source = source

        # Initialize the optimizer.
        optimizer = self._init_optimizer(optimizer_type, lr)

        # Initialize variables to store training performance.
        epoch_history, training_loss_history, stopping_triggered = [], [], False
        if early_stopping:
            validation_loss_history = []
        times = {
            "total": 0.0,
            "preparation": 0.0,
            "training": 0.0,
            "overhead": 0.0,
            "forward": 0.0,
            "loss": 0.0,
            "backward": 0.0,
            "unaccounted": 0.0,
        }

        # Print the configuration.
        if verbose:
            print("Configuration:")
            print("=" * len("Configuration:"))
            print(f"Source points:       {source.shape[0]}")
            print(f"Target points:       {target.shape[0]}")
            print(f"Epochs:              {epochs}")
            print(f"Optimizer:           {optimizer_type}")
            print(f"Learning rate:       {lr}")
            print(f"Batch size:          {batch_size}")
            print(f"Early stopping:      {early_stopping}")
            if early_stopping:
                print(f"Validation fraction: {validation_fraction}")
                print(f"Patience:            {patience}")
                print(f"Min. epochs:         {min_epochs}")
        print()

        times["preparation"] = time.time() - prep_start
        self.train()
        try:
            # Training loop.
            training_start = time.time()
            for epoch in range(epochs):
                # Train the model for one epoch.
                training_loss_current, times_current = self._training_loop(
                    training_source,
                    target,
                    device,
                    loss_function,
                    optimizer,
                    batch_size,
                )

                # Update training performance.
                epoch_history.append(epoch)
                training_loss_history.append(training_loss_current)
                for key in ["overhead", "forward", "loss", "backward"]:
                    times[key] += times_current[key]

                # Evaluate the model on the validation set.
                if early_stopping:
                    # Compute the validation loss.
                    validation_loss_current, times_current = self._validation_loop(
                        validation_source,
                        target,
                        device,
                        loss_function,
                        optimizer,
                        batch_size,
                    )

                    # Update training performance.
                    validation_loss_history.append(validation_loss_current)
                    for key in ["overhead", "forward", "loss"]:
                        times[key] += times_current[key]

                    # Update the progress bar.
                    DisplayUtils.print_progress_bar(
                        epoch + 1,
                        epochs,
                        training_loss_current,
                        validation_loss_current,
                        None,
                        bar_length=40,
                    )

                    # Check early stopping.
                    if NeuralNetwork._early_stopping(
                        validation_loss_history, patience, min_epochs
                    ):
                        stopping_triggered = True
                        print(f"\n\nEarly stopping triggered at epoch {epoch+1}.")
                        break
                else:
                    # Update the progress bar.
                    DisplayUtils.print_progress_bar(
                        epoch + 1,
                        epochs,
                        training_loss_current,
                        None,
                        None,
                        bar_length=40,
                    )

            # Compute the total training time.
            times["training"] = time.time() - training_start
            times["total"] = times["preparation"] + times["training"]

            # Print the results.
            if verbose:
                DisplayUtils.print_training_results(
                    times, len(epoch_history), stopping_triggered
                )

            # Plot the results.
            if plot_results and early_stopping:
                DisplayUtils.plot_loss_and_lr(
                    epoch_history, training_loss_history, validation_loss_history
                )
            elif plot_results and (not early_stopping):
                DisplayUtils.plot_loss_and_lr(
                    epoch_history, training_loss_history, None
                )

        finally:
            self.zero_grad()
            self.eval()
            NeuralNetwork._clear_cache(device)
            gc.collect()

    def evaluate_model(
        self,
        source: pv.PolyData | torch.Tensor,
        device: torch.device,
        batch_size: int = 1024,
        clear_cache: bool = True,
        clear_every: int = 10,
    ) -> np.ndarray | torch.Tensor:
        """
        Evaluate model
        ==============

        Predicts the deformation field that maps the source mesh to the target mesh specified
        during training.
        """
        self.eval()

        with torch.no_grad():
            # Prepare source tensor from input
            source = PreparationUtils.prepare_source(self, source, device)

            # Compute the deformation in batches.
            deformations = []
            for batch_start in range(0, source.shape[0], batch_size):
                # Compute the batch deformation.
                batch_deformation = self(
                    source[batch_start : min(batch_start + batch_size, source.shape[0])]
                )

                # Append the batch deformation to the deformations array.
                deformations.append(batch_deformation.cpu())

                # Periodically clear the GPU cache.
                if (
                    (batch_start // batch_size) % clear_every == 0
                    and batch_start > 0
                    and clear_cache
                ):
                    NeuralNetwork._clear_cache(device)

            if clear_cache:
                NeuralNetwork._clear_cache(device)
                gc.collect()

            # Concatenate the deformations.
            return torch.cat(deformations, dim=0).numpy()

    @staticmethod
    def _clear_cache(device: torch.device):
        """
        Clear cache
        ===========

        Clears the GPU cache.
        """
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _early_stopping(
        validation_loss_history: list[float], patience: int, min_epochs: int
    ) -> bool:
        """
        Early stopping
        =================

        Checks if training should stop early.
        """
        if len(validation_loss_history) < min_epochs:
            return False
        if len(validation_loss_history) < patience + 1:
            return False

        # Find how long ago the best loss was.
        best_loss_idx = validation_loss_history.index(min(validation_loss_history))
        current_idx = len(validation_loss_history) - 1

        # Stop if the best loss was too long ago.
        if (current_idx - best_loss_idx) >= patience:
            return True

    def _init_optimizer(self, optimizer_type: str, lr: float) -> torch.optim.Optimizer:
        """
        Initialize optimizer
        ====================

        Initializes and returns the optimizer.
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

        if optimizer_type not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_type}.")
        return optimizers[optimizer_type](self.parameters(), lr=lr)

    def _lr_test_loop(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        device: torch.device,
        loss_function: LossFunctionProtocol,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        batch_size: int,
    ):
        """
        Learning rate test loop
        =======================

        Trains the model for one iteration of the learning rate test.
        """
        # Set the learning rate for this iteration.
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        # Generate source and target batches.
        source_batches, target_batches = PreparationUtils.compute_epoch_batches(
            source, target, batch_size, batch_size, device
        )
        batch_source, batch_target = source_batches[0], target_batches[0]

        # Compute the deformation.
        optimizer.zero_grad()
        batch_deformation = self(batch_source)

        # Compute the loss.
        batch_loss = loss_function(batch_source[:, :3], batch_target, batch_deformation)

        # Backpropagate the loss.
        batch_loss.backward()
        optimizer.step()

        return batch_loss.item()

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

    def _training_loop(
        self,
        training_source: torch.Tensor,
        target: torch.Tensor,
        device: torch.device,
        loss_function: LossFunctionProtocol,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
    ):
        """
        Training loop
        =============

        Trains the model for one epoch.
        """
        loss = 0
        times = {
            "overhead": 0.0,
            "forward": 0.0,
            "loss": 0.0,
            "backward": 0.0,
        }

        # Extract source and target batches.
        overhead_start = time.time()
        source_batches, target_batches = PreparationUtils.compute_epoch_batches(
            training_source,
            target,
            batch_size,
            batch_size,
            device,
        )
        max_num_batches = max(len(source_batches), len(target_batches))

        # Batches loop.
        for batch_idx in range(max_num_batches):
            # Sample a subset of the source and target.
            batch_source = source_batches[batch_idx % len(source_batches)]
            batch_target = target_batches[batch_idx % len(target_batches)]
            self._sync(device)
            times["overhead"] += time.time() - overhead_start

            # Compute the deformation.
            forward_start = time.time()
            optimizer.zero_grad()
            batch_deformation = self(batch_source)
            self._sync(device)
            times["forward"] += time.time() - forward_start

            # Compute the loss.
            loss_start = time.time()
            batch_loss = loss_function(
                batch_source[:, :3], batch_target, batch_deformation
            )
            self._sync(device)
            times["loss"] = time.time() - loss_start
            loss += batch_loss.item()

            # Backpropagate the loss.
            backward_start = time.time()
            batch_loss.backward()
            optimizer.step()
            self._sync(device)
            times["backward"] += time.time() - backward_start

            overhead_start = time.time()

        # Compute the loss and learning rate.
        loss = loss / max_num_batches
        return loss, times

    def _validation_loop(
        self,
        validation_source: torch.Tensor,
        target: torch.Tensor,
        device: torch.device,
        loss_function: LossFunctionProtocol,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
    ):
        """
        Validation loop
        ===============

        Computes the loss on the validation set.
        """
        loss = 0
        times = {
            "overhead": 0.0,
            "forward": 0.0,
            "loss": 0.0,
        }
        # Extract source and target batches.
        overhead_start = time.time()
        source_batches, target_batches = PreparationUtils.compute_epoch_batches(
            validation_source,
            target,
            batch_size,
            batch_size,
            device,
        )
        max_num_batches = max(len(source_batches), len(target_batches))

        # Batches loop.
        for batch_idx in range(max_num_batches):
            # Sample a subset of the source and target.
            batch_source = source_batches[batch_idx % len(source_batches)]
            batch_target = target_batches[batch_idx % len(target_batches)]
            self._sync(device)
            times["overhead"] += time.time() - overhead_start

            # Compute the deformation.
            forward_start = time.time()
            optimizer.zero_grad()
            batch_deformation = self(batch_source)
            self._sync(device)
            times["forward"] += time.time() - forward_start

            # Compute the loss.
            loss_start = time.time()
            batch_loss = loss_function(
                batch_source[:, :3], batch_target, batch_deformation
            )
            self._sync(device)
            times["loss"] = time.time() - loss_start
            loss += batch_loss.item()

            overhead_start = time.time()

        # Compute the loss.
        loss = loss / max_num_batches
        return loss, times


class LossFunction:
    """
    Loss function
    =============

    This class groups functions related to loss functions.
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


class PreparationUtils:
    """
    Preparation utils
    =================

    This class groups utility functions related to preparing data for training.
    """

    @staticmethod
    def prepare_source(
        model: NeuralNetwork, source: pv.PolyData | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Prepare source
        ==============

        Converts a source mesh or tensor to the appropriate format for the model.
        """
        # Error handling.
        if isinstance(source, pv.PolyData) and (
            source.points is None or len(source.points) == 0
        ):
            raise ValueError("source cannot be empty.")
        if (
            isinstance(source, pv.PolyData)
            and model.input_dim == 6
            and ("Normals" not in source.point_data)
        ):
            source = source.compute_normals()
            if "Normals" not in source.point_data:
                raise ValueError("Failed to compute normals for mesh.")
        if isinstance(source, torch.Tensor) and source.ndim != 2:
            raise ValueError("source must be 2D.")
        if isinstance(source, torch.Tensor) and source.shape[0] == 0:
            raise ValueError("source cannot be empty.")
        if isinstance(source, torch.Tensor) and source.shape[1] != model.input_dim:
            raise ValueError(f"source must have shape (N, {model.input_dim}).")

        # Prepare the source.
        if isinstance(source, pv.PolyData):
            points = torch.from_numpy(source.points).float().to(device)
            if model.input_dim == 3:
                return points
            elif model.input_dim == 6:
                normals = (
                    torch.from_numpy(source.point_data["Normals"]).float().to(device)
                )
                return torch.cat((points, normals), dim=1)
            else:
                raise ValueError("model.input_dim must be 3 or 6.")
        elif isinstance(source, torch.Tensor):
            return source.to(device)
        else:
            raise TypeError("source must be pv.PolyData or torch.Tensor.")

    @staticmethod
    def prepare_target(
        model: NeuralNetwork, target: pv.PolyData | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Prepare target
        ==============

        Converts a target mesh or tensor to the appropriate format for the model.
        """
        # Error handling.
        if isinstance(target, pv.PolyData) and (
            target.points is None or len(target.points) == 0
        ):
            raise ValueError("target cannot be empty.")
        if isinstance(target, torch.Tensor) and target.ndim != 2:
            raise ValueError("target must be 2D.")
        if isinstance(target, torch.Tensor) and target.shape[0] == 0:
            raise ValueError("target cannot be empty.")
        if isinstance(target, torch.Tensor) and target.shape[1] != model.output_dim:
            raise ValueError(f"target must have shape (N, {model.output_dim}).")

        # Prepare the target.
        if isinstance(target, pv.PolyData):
            points = torch.from_numpy(target.points).float().to(device)
            if model.output_dim == 3:
                return points
            else:
                raise ValueError("model.input_dim must be 3 or 6.")
        elif isinstance(target, torch.Tensor):
            return target.to(device)
        else:
            raise TypeError("target must be pv.PolyData or torch.Tensor.")

    @staticmethod
    def split_source(source: torch.Tensor, validation_fraction: float):
        """
        Split source
        ============

        Splits the source into a training set and a validation set.
        """
        # Calculate the number of validation samples
        num_validation = int(validation_fraction * source.shape[0])

        if num_validation == 0:
            raise ValueError("Not enough points to make a validation set.")
        else:
            indices = torch.randperm(source.shape[0], device=source.device)
            validation_indices = indices[:num_validation]
            training_indices = indices[num_validation:]
            return source[training_indices], source[validation_indices]

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


class DisplayUtils:
    """
    Display utils
    =============

    This class groups functions related to displaying training results.
    """

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
        _, ax1 = plt.subplots(1, 1, figsize=(8, 5))

        # Plot loss vs learning rate on a log scale.
        ax1.set_xlabel("Learning rate")
        ax1.set_xscale("log")
        ax1.set_ylabel("Loss")
        ax1.set_yscale("log")
        ax1.grid(True, which="both", axis="both", alpha=0.3)
        ax1.plot(learning_rates, losses, alpha=0.6, label="Raw Loss")
        ax1.plot(learning_rates, smoothed_losses, linewidth=2, label="Smoothed Loss")
        ax1.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_progress_bar(
        epoch: int,
        epochs: int,
        training_loss: float,
        validation_loss: float | None,
        lr: float | None,
        bar_length: int,
    ):
        """
        Print progress bar
        ==================

        Displays a progress bar.
        """
        filled_length = int(bar_length * epoch / epochs)
        progress_bar = "█" * filled_length + "-" * (bar_length - filled_length)

        if validation_loss is None and lr is None:
            print(
                f"\r[{progress_bar}] {epoch}/{epochs} ({(epoch/epochs):.1%}) | "
                f"Loss: {training_loss:.6f}.",
                end="",
                flush=True,
            )
        elif validation_loss is not None and lr is None:
            print(
                f"\r[{progress_bar}] {epoch}/{epochs} ({(epoch/epochs):.1%}) | "
                f"Training loss: {training_loss:.6f} | "
                f"Validation loss: {validation_loss:.6f}.",
                end="",
                flush=True,
            )
        elif validation_loss is None and lr is not None:
            print(
                f"\r[{progress_bar}] {epoch}/{epochs} ({(epoch/epochs):.1%}) | "
                f"Loss: {training_loss:.6f} | "
                f"Learning rate: {lr:.3e}.",
                end="",
                flush=True,
            )
        else:
            print(
                f"\r[{progress_bar}] {epoch}/{epochs} ({(epoch/epochs):.1%}) | "
                f"Training loss: {training_loss:.6f} | "
                f"Validation loss: {validation_loss:.6f} | "
                f"Learning rate: {lr:.3e}.",
                end="",
                flush=True,
            )

    @staticmethod
    def print_training_results(times: dict, epochs: int, stopping_triggered: bool):
        """
        Print training results
        ======================

        Prints the results of training.
        """
        # Compute the accounted and unaccounted time
        accounted_time = 0
        for key in ["preparation", "overhead", "forward", "loss", "backward"]:
            accounted_time += times[key]
        times["unaccounted"] = times["total"] - accounted_time

        # Convert the times to fractions.
        fractions = {
            key: times[key] / times["total"] for key in times if key != "total"
        }

        # Print the times.
        if stopping_triggered:
            print("\nResults:")
        else:
            print("\n\nResults:")
        print("=" * len("Results:"))
        print(f"Training completed after {epochs} epochs.")
        print(f"Total time:  {times["total"]:.2f}s")
        print(f"Preparation: {fractions["preparation"]:.1%}.")
        print(f"Training:    {fractions["training"]:.1%}.")
        print(f"Overhead:    {fractions["overhead"]:.1%}.")
        print(f"Forward:     {fractions["forward"]:.1%}.")
        print(f"Loss:        {fractions["loss"]:.1%}.")
        print(f"Backward:    {fractions["backward"]:.1%}.")
        print(f"Unaccounted: {fractions["unaccounted"]:.1%}.")

    @staticmethod
    def plot_loss_and_lr(
        epoch_history: list[int],
        training_loss_history: list[float],
        validation_loss_history: list[float] | None,
    ):
        """
        Plot loss and learning rate
        ===========================

        Plots the loss and learning rate upon completion of training.
        """
        if validation_loss_history is None:
            # Initialize the plot.
            _, ax1 = plt.subplots(1, 1, figsize=(8, 5))

            # Plot the training loss (log scale).
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_yscale("log")
            ax1.grid(True, which="both", axis="both", alpha=0.3)
            ax1.scatter(epoch_history, training_loss_history, s=10)
        else:
            # Initialize the plot.
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

            # Plot the training loss (log scale).
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Training loss")
            ax1.set_yscale("log")
            ax1.grid(True, which="both", axis="both", alpha=0.3)
            ax1.scatter(epoch_history, training_loss_history, s=10)

            # Plot the validation loss (log scale).
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Validation loss")
            ax2.set_yscale("log")
            ax2.grid(True, which="both", axis="both", alpha=0.3)
            ax2.scatter(epoch_history, validation_loss_history, s=10)

        plt.tight_layout()
        plt.show()
