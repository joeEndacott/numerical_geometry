"""
Utils
=====

This module contains a range of useful utility functions.
"""

import numpy as np
import pyvista as pv


def pyvista_faces_to_numpy(faces: np.ndarray):
    """
    PyVista faces to numpy
    ======================

    Converts a PyVista triangles array to a numpy array of shape (n_faces, 3).
    Assumes all faces are triangles (first value in each group is 3).
    """
    # Error checking.
    if faces.ndim != 1:
        raise ValueError("faces array must be one-dimensional")

    if len(faces) % 4 != 0:
        raise ValueError(
            "faces array length must be divisible by 4 for triangular faces"
        )

    if len(faces) == 0:
        raise ValueError("faces array cannot be empty")

    n_faces = len(faces) // 4
    return faces.reshape((n_faces, 4))[:, 1:]


def numpy_faces_to_pyvista(faces: np.ndarray):
    """
    Numpy faces to PyVista
    ======================

    Converts a numpy triangles array of shape (n_faces, 3) to the format expected by PyVista.
    """
    # Error checking.
    if faces.ndim != 2:
        raise ValueError("faces array must be two-dimensional")

    if faces.shape[1] != 3:
        raise ValueError("faces array must have shape (n_faces, 3)")

    if len(faces) == 0:
        raise ValueError("faces array cannot be empty")

    n_faces = faces.shape[0]
    return np.hstack([np.full((n_faces, 1), 3, dtype=int), faces]).ravel()


class Sphere:
    """
    Sphere
    ======

    This class groups methods related to generating a sphere mesh.
    """

    @staticmethod
    def _add_pole_faces(mesh: pv.PolyData, num_azimuthal_angles: int):
        """
        Add pole faces
        ==============

        Appends all of the faces involving the poles to the mesh.
        This function assumes that `_wrap_phi` has not been called.
        """
        # Extract the faces from the PyVista mesh.
        faces = pyvista_faces_to_numpy(mesh.faces)
        max_vertex = np.max(faces)

        # Remove all faces involving the poles.
        bulk_faces = faces[
            ~((faces == 0).any(axis=1) | (faces == np.max(faces)).any(axis=1))
        ]

        # Generate faces made by the North pole.
        north_faces = np.zeros((num_azimuthal_angles - 1, 3), dtype=int)
        for i in range(num_azimuthal_angles - 1):
            north_faces[i, :] = np.array([i + 2, 0, i + 1])

        # Generate faces made by the South pole.
        south_faces = np.zeros((num_azimuthal_angles - 1, 3), dtype=int)
        for i in range(num_azimuthal_angles - 1):
            south_faces[i, :] = np.array(
                [
                    max_vertex,
                    max_vertex - num_azimuthal_angles + i + 1,
                    max_vertex - num_azimuthal_angles + i,
                ]
            )

        return pv.PolyData(
            mesh.points,
            numpy_faces_to_pyvista(np.vstack((north_faces, bulk_faces, south_faces))),
        )

    @staticmethod
    def _wrap_phi(mesh: pv.PolyData, num_polar_angles: int, num_azimuthal_angles: int):
        """
        Wrap phi
        ========

        This function connects the points with the largest phi values to the points phi = 0.
        This function assumes that `_add_pole_faces` has already been called.
        """
        # Extract the faces from the PyVista mesh.
        faces = pyvista_faces_to_numpy(mesh.faces)
        max_vertex = np.max(faces)

        # Wrap the phi coordinate in the bulk.
        wrapping_faces = np.zeros((2 * (num_polar_angles - 1) + 2, 3), dtype=int)
        for i in range(num_polar_angles - 1):
            # Make the first triangle.
            wrapping_faces[2 * i, :] = np.array(
                [
                    (i + 2) * num_azimuthal_angles,
                    (i * num_azimuthal_angles) + 1,
                    (i + 1) * num_azimuthal_angles,
                ]
            )

            # Make the second triangle.
            wrapping_faces[(2 * i) + 1, :] = np.array(
                [
                    (i + 2) * num_azimuthal_angles,
                    (i + 1) * num_azimuthal_angles + 1,
                    (i * num_azimuthal_angles) + 1,
                ]
            )

        # Wrap the phi coordinates at the poles.
        wrapping_faces[-2, :] = np.array([1, num_azimuthal_angles, 0])
        wrapping_faces[-1, :] = np.array(
            [max_vertex, max_vertex - 1, max_vertex - num_azimuthal_angles]
        )

        return pv.PolyData(
            mesh.points, numpy_faces_to_pyvista(np.vstack((faces, wrapping_faces)))
        )

    @staticmethod
    def _correct_mesh(mesh, num_polar_angles, num_azimuthal_angles):
        """
        Correct mesh
        ============

        Corrects the tessellation produced by the `delaunay_2d()` function, and returns a new mesh.
        """
        mesh_1 = Sphere._add_pole_faces(mesh, num_azimuthal_angles)
        mesh_2 = Sphere._wrap_phi(mesh_1, num_polar_angles, num_azimuthal_angles)
        return mesh_2

    @staticmethod
    def make_sphere(
        num_polar_angles: int,
        num_azimuthal_angles: int,
        centre: np.ndarray,
        radius: float,
    ):
        """
        Make sphere
        ===========

        Generates a mesh of a sphere with a specified number of polar and azimuthal angles.
        """
        # Generate arrays of theta, phi values for all points, excluding the poles.
        theta = np.linspace(0, np.pi, num_polar_angles)[1:-1]
        phi = np.linspace(0, 2 * np.pi, num_azimuthal_angles + 1)[:-1]

        # Make an array of vertices for the bulk of the sphere.
        r_v, theta_v, phi_v = np.meshgrid(np.array([radius]), theta, phi)
        sphere_vertices_spherical_no_poles = np.c_[
            r_v.reshape(-1), theta_v.reshape(-1), phi_v.reshape(-1)
        ]

        # Add the poles to the array of vertices.
        sphere_vertices_spherical = np.zeros(
            (sphere_vertices_spherical_no_poles.shape[0] + 2, 3), dtype=float
        )
        sphere_vertices_spherical[0, :] = [radius, 0.0, 0.0]
        sphere_vertices_spherical[1:-1, :] = sphere_vertices_spherical_no_poles
        sphere_vertices_spherical[-1, :] = [radius, np.pi, 0.0]

        # Create a PyVista mesh, and perform a Delaunay 2D tessellation.
        sphere_mesh_spherical_delaunay = pv.PolyData(
            sphere_vertices_spherical
        ).delaunay_2d()

        # Correct problems with the tessellation.
        sphere_mesh_spherical_custom = Sphere._add_pole_faces(
            sphere_mesh_spherical_delaunay, len(phi)
        )
        sphere_mesh_spherical_custom = Sphere._wrap_phi(
            sphere_mesh_spherical_custom, len(theta), len(phi)
        )

        # Make an array storing the vertices in Cartesian coordinates.
        x_v = r_v * np.sin(theta_v) * np.cos(phi_v)
        y_v = r_v * np.sin(theta_v) * np.sin(phi_v)
        z_v = r_v * np.cos(theta_v)
        sphere_vertices_cartesian_no_poles = np.c_[
            x_v.reshape(-1), y_v.reshape(-1), z_v.reshape(-1)
        ]

        # Add the poles to the array of vertices.
        sphere_vertices_cartesian = np.zeros(
            (sphere_vertices_cartesian_no_poles.shape[0] + 2, 3), dtype=float
        )
        sphere_vertices_cartesian[0, :] = [0.0, 0.0, radius]
        sphere_vertices_cartesian[1:-1, :] = sphere_vertices_cartesian_no_poles
        sphere_vertices_cartesian[-1, :] = [0.0, 00, -radius]

        # Add an offset to the vertices.
        sphere_vertices_cartesian += centre

        # Create a new mesh, using the triangles array generated for the 2D mesh.
        return pv.PolyData(
            sphere_vertices_cartesian, sphere_mesh_spherical_custom.faces
        )


def sphere(
    num_polar_angles: int = 20,
    num_azimuthal_angles: int = 40,
    centre: np.ndarray = np.array([0, 0, 0]),
    radius: float = 1.0,
):
    """
    Sphere
    ======

    Generates a mesh of a sphere with a specified number of polar and azimuthal angles.
    """
    return Sphere.make_sphere(num_polar_angles, num_azimuthal_angles, centre, radius)


class Cube:
    """
    Cube
    ====

    This class groups methods related to generating a cube mesh.
    """

    @staticmethod
    def make_cube(num_points_per_side: int, side_length: float, centre: np.ndarray):
        """
        Make cube
        =========

        Generates a mesh a cube with a specified number of points per side.
        """
        # Make an array of vertices for a square.
        x = np.linspace(-side_length / 2, side_length / 2, num_points_per_side)
        y = np.linspace(-side_length / 2, side_length / 2, num_points_per_side)
        z = np.array([0])
        x_v, y_v, z_v = np.meshgrid(x, y, z)
        square_vertices_2d = np.c_[x_v.reshape(-1), y_v.reshape(-1), z_v.reshape(-1)]

        # Generate vertices and triangles for the x = -L/2 face.
        vertices_x_min = np.zeros(square_vertices_2d.shape)
        vertices_x_min[:, 0] = -side_length / 2
        vertices_x_min[:, 1] = square_vertices_2d[:, 1]
        vertices_x_min[:, 2] = square_vertices_2d[:, 0]
        faces_x_min = pyvista_faces_to_numpy(
            pv.PolyData(vertices_x_min).delaunay_2d().faces
        )

        # Generate vertices and triangles for the x = +L/2 face.
        vertices_x_max = np.zeros(square_vertices_2d.shape)
        vertices_x_max[:, 0] = +side_length / 2
        vertices_x_max[:, 1] = square_vertices_2d[:, 1]
        vertices_x_max[:, 2] = square_vertices_2d[:, 0]
        faces_x_max = pyvista_faces_to_numpy(
            pv.PolyData(vertices_x_max).delaunay_2d().faces
        )

        # Generate vertices and triangles for the y = -L/2 face.
        vertices_y_min = np.zeros(square_vertices_2d.shape)
        vertices_y_min[:, 0] = square_vertices_2d[:, 0]
        vertices_y_min[:, 1] = -side_length / 2
        vertices_y_min[:, 2] = square_vertices_2d[:, 1]
        faces_y_min = pyvista_faces_to_numpy(
            pv.PolyData(vertices_y_min).delaunay_2d().faces
        )

        # Generate vertices and triangles for the y = +L/2 face.
        vertices_y_max = np.zeros(square_vertices_2d.shape)
        vertices_y_max[:, 0] = square_vertices_2d[:, 0]
        vertices_y_max[:, 1] = side_length / 2
        vertices_y_max[:, 2] = square_vertices_2d[:, 1]
        faces_y_max = pyvista_faces_to_numpy(
            pv.PolyData(vertices_y_max).delaunay_2d().faces
        )

        # Generate vertices and triangles for the z = -L/2 face.
        vertices_z_min = np.zeros(square_vertices_2d.shape)
        vertices_z_min[:, 0] = square_vertices_2d[:, 1]
        vertices_z_min[:, 1] = square_vertices_2d[:, 0]
        vertices_z_min[:, 2] = -side_length / 2
        faces_z_min = pyvista_faces_to_numpy(
            pv.PolyData(vertices_z_min).delaunay_2d().faces
        )

        # Generate vertices and triangles for the z = +L/2 face.
        vertices_z_max = np.zeros(square_vertices_2d.shape)
        vertices_z_max[:, 0] = square_vertices_2d[:, 1]
        vertices_z_max[:, 1] = square_vertices_2d[:, 0]
        vertices_z_max[:, 2] = side_length / 2
        faces_z_max = pyvista_faces_to_numpy(
            pv.PolyData(vertices_z_max).delaunay_2d().faces
        )

        # Create an array of vertices for the cube.
        cube_vertices_cartesian = np.vstack(
            (
                vertices_x_min,
                vertices_x_max,
                vertices_y_min,
                vertices_y_max,
                vertices_z_min,
                vertices_z_max,
            )
        )
        cube_vertices_cartesian += centre

        # Create an array of faces.
        num_vertices = square_vertices_2d.shape[0]
        cube_faces = numpy_faces_to_pyvista(
            np.vstack(
                (
                    faces_x_min,
                    faces_x_max + num_vertices,
                    faces_y_min + 2 * num_vertices,
                    faces_y_max + 3 * num_vertices,
                    faces_z_min + 4 * num_vertices,
                    faces_z_max + 5 * num_vertices,
                )
            )
        )

        # Create a 3D mesh of the cube.
        cube_mesh_cartesian_custom = pv.PolyData(cube_vertices_cartesian, cube_faces)
        return cube_mesh_cartesian_custom.clean()


def cube(
    num_points_per_side: int = 20,
    side_length: float = 2,
    centre: np.ndarray = np.array([0, 0, 0]),
):
    """
    Cube
    ====

    Generates a mesh of a cube with a specified number of points per side.
    """
    return Cube.make_cube(num_points_per_side, side_length, centre)
