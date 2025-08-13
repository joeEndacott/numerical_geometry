import numpy as np


def pyvista_faces_to_numpy(faces):
    """
    PyVista faces to numpy
    ======================

    Converts a PyVista triangles array to a numpy array of shape (n_faces, 3).
    Assumes all faces are triangles (first value in each group is 3).
    """
    n_faces = len(faces) // 4
    return faces.reshape((n_faces, 4))[:, 1:]


def numpy_faces_to_pyvista(faces):
    """
    Numpy faces to PyVista
    ======================

    Converts a numpy triangles array of shape (n_faces, 3) to the format expected by PyVista.
    """
    n_faces = faces.shape[0]
    return np.hstack([np.full((n_faces, 1), 3, dtype=int), faces]).ravel()
