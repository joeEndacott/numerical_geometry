"""
Numerical geometry
==================

A numerical geometry project in Python, with a focus on mesh deformations using neural-networks.
"""

# Import the core module.
from . import core

# Import key classes.
from .core import Geometry, NeuralNetwork, RayTracing

# Import key functions.
sphere = core.Geometry.sphere
cube = core.Geometry.cube
get_chamfer_distance = core.Geometry.get_chamfer_distance
get_average_deformation = core.Geometry.get_average_deformation
animate_deformation = core.Geometry.animate_deformation
get_intersection_points = core.RayTracing.get_intersection_points


# Version info.
__version__ = "0.1.0"

# Define what gets imported with "from numerical_geometry import *".
__all__ = ["core", "sphere", "cube", "animate_deformation", "Geometry", "NeuralNetwork"]
