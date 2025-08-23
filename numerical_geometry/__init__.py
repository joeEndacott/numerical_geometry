"""
Numerical geometry
==================

A numerical geometry project in Python, with a focus on mesh deformations using neural-networks.
"""

# Import the modules and key classes.
from . import geometry, neural_network
from .neural_network import NeuralNetwork

# Import key functions from geometry.
sphere = geometry.Sphere.sphere
cube = geometry.Cube.cube
animate_deformation = geometry.Utils.animate_deformation

# Import key functions from neural_network.
chamfer_distance = neural_network.LossFunction.chamfer_distance
deformation_loss = neural_network.LossFunction.deformation_loss
laplacian_loss = neural_network.LossFunction.laplacian_loss
edge_length_loss = neural_network.LossFunction.edge_length_loss


# Version info.
__version__ = "0.1.0"

# Define what gets imported with "from numerical_geometry import *".
__all__ = [
    "geometry",
    "neural_network",
    "sphere",
    "cube",
    "animate_deformation",
    "NeuralNetwork",
    "chamfer_distance",
    "deformation_loss",
    "laplacian_loss",
]
