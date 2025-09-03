"""
Geometry
========

This module groups functions related to 3D mesh generation and manipulation using PyVista.
"""

from IPython.display import HTML
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Sphere:
    """
    Sphere
    ======

    This class groups functions related to generating a sphere using PyVista.
    """

    @staticmethod
    def _add_pole_faces(mesh: pv.PolyData, num_azimuthal_angles: int) -> pv.PolyData:
        """
        Add pole faces
        ==============

        Appends all of the faces involving the poles to the mesh.
        This function assumes that `_wrap_phi` has not been called.
        """
        # Extract the faces from the PyVista mesh.
        faces = FacesUtils.pyvista_faces_to_numpy(mesh.faces)
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
            FacesUtils.numpy_faces_to_pyvista(
                np.vstack((north_faces, bulk_faces, south_faces))
            ),
        )

    @staticmethod
    def _wrap_phi(
        mesh: pv.PolyData, num_polar_angles: int, num_azimuthal_angles: int
    ) -> pv.PolyData:
        """
        Wrap phi
        ========

        This function connects the points with the largest phi values to the points phi = 0.
        This function assumes that `_add_pole_faces` has already been called.
        """
        # Extract the faces from the PyVista mesh.
        faces = FacesUtils.pyvista_faces_to_numpy(mesh.faces)
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
            mesh.points,
            FacesUtils.numpy_faces_to_pyvista(np.vstack((faces, wrapping_faces))),
        )

    @staticmethod
    def sphere(
        num_polar_angles: int = 20,
        num_azimuthal_angles: int = 40,
        centre: np.ndarray = np.array([0, 0, 0]),
        radius: float = 1.0,
    ) -> pv.PolyData:
        """
        Sphere
        ======

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


class Cube:
    """
    Cube
    ====

    This class groups methods related to generating a sphere using PyVista.
    """

    @staticmethod
    def cube(
        num_points_per_side: int = 2,
        side_length: float = 2.0,
        centre: np.ndarray = np.array([0, 0, 0]),
    ) -> pv.PolyData:
        """
        Cube
        ====

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
        faces_x_min = FacesUtils.pyvista_faces_to_numpy(
            pv.PolyData(vertices_x_min).delaunay_2d().faces
        )

        # Generate vertices and triangles for the x = +L/2 face.
        vertices_x_max = np.zeros(square_vertices_2d.shape)
        vertices_x_max[:, 0] = +side_length / 2
        vertices_x_max[:, 1] = square_vertices_2d[:, 1]
        vertices_x_max[:, 2] = square_vertices_2d[:, 0]
        faces_x_max = FacesUtils.pyvista_faces_to_numpy(
            pv.PolyData(vertices_x_max).delaunay_2d().faces
        )

        # Generate vertices and triangles for the y = -L/2 face.
        vertices_y_min = np.zeros(square_vertices_2d.shape)
        vertices_y_min[:, 0] = square_vertices_2d[:, 0]
        vertices_y_min[:, 1] = -side_length / 2
        vertices_y_min[:, 2] = square_vertices_2d[:, 1]
        faces_y_min = FacesUtils.pyvista_faces_to_numpy(
            pv.PolyData(vertices_y_min).delaunay_2d().faces
        )

        # Generate vertices and triangles for the y = +L/2 face.
        vertices_y_max = np.zeros(square_vertices_2d.shape)
        vertices_y_max[:, 0] = square_vertices_2d[:, 0]
        vertices_y_max[:, 1] = side_length / 2
        vertices_y_max[:, 2] = square_vertices_2d[:, 1]
        faces_y_max = FacesUtils.pyvista_faces_to_numpy(
            pv.PolyData(vertices_y_max).delaunay_2d().faces
        )

        # Generate vertices and triangles for the z = -L/2 face.
        vertices_z_min = np.zeros(square_vertices_2d.shape)
        vertices_z_min[:, 0] = square_vertices_2d[:, 1]
        vertices_z_min[:, 1] = square_vertices_2d[:, 0]
        vertices_z_min[:, 2] = -side_length / 2
        faces_z_min = FacesUtils.pyvista_faces_to_numpy(
            pv.PolyData(vertices_z_min).delaunay_2d().faces
        )

        # Generate vertices and triangles for the z = +L/2 face.
        vertices_z_max = np.zeros(square_vertices_2d.shape)
        vertices_z_max[:, 0] = square_vertices_2d[:, 1]
        vertices_z_max[:, 1] = square_vertices_2d[:, 0]
        vertices_z_max[:, 2] = side_length / 2
        faces_z_max = FacesUtils.pyvista_faces_to_numpy(
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
        cube_faces = FacesUtils.numpy_faces_to_pyvista(
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


class RayTracing:
    """
    Ray tracing
    ===========

    This class groups functions related to ray tracing deformation methods.
    """

    @staticmethod
    def get_intersection_points(
        source_mesh: pv.PolyData, target_mesh: pv.PolyData, ray_length: float = 10.0
    ):
        """
        Get intersection points
        =======================

        Calculates where the vertex normals to the source mesh intersect the faces of the target
        mesh. The intersection points are calculated using the PyVista `ray_trace()` method.
        """
        intersection_points = []
        intersection_rays = []
        intersection_cells = []

        # Extract vertices and normals from the sphere mesh.
        sphere_vertices = source_mesh.points
        sphere_normals = source_mesh.point_data["Normals"]

        # Process each ray individually.
        for i, (origin, normal) in enumerate(zip(sphere_vertices, sphere_normals)):
            # Perform ray tracing.
            try:
                end_point = origin + normal * ray_length
                points, cells = target_mesh.ray_trace(
                    origin, end_point, first_point=True
                )

                # If intersection found, store the results
                if len(points) > 0:
                    intersection_points.append(points)
                    intersection_rays.append(i)
                    intersection_cells.append(cells)

            except (RuntimeError, ValueError, IndexError) as _:
                continue

        # Convert results to numpy arrays.
        intersection_points = (
            np.array(intersection_points) if intersection_points else np.empty((0, 3))
        )
        intersection_rays = (
            np.array(intersection_rays)
            if intersection_rays
            else np.empty((0,), dtype=int)
        )
        intersection_cells = (
            np.array(intersection_cells)
            if intersection_cells
            else np.empty((0,), dtype=int)
        )

        return intersection_points, intersection_rays, intersection_cells


class FacesUtils:
    """
    Utils
    =====

    This class groups utility functions related to mesh faces.
    """

    @staticmethod
    def pyvista_faces_to_numpy(faces: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def numpy_faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def get_edges(mesh: pv.PolyData) -> np.ndarray:
        """
        Get edges
        =========

        Extracts the edges from a PyVista mesh, and turns them into an (num_edges, 2) NumPy array
        of vertex indices.
        """
        edges = mesh.extract_all_edges()
        return edges.lines.reshape(-1, 3)[:, 1:]


class PlottingUtils:
    """
    Plotting utils
    ==============

    This class groups utility functions related to plotting.
    """

    @staticmethod
    def plot_source_and_target(source_mesh: pv.PolyData, target_mesh: pv.PolyData):
        """
        Plot source and target
        ======================

        Plots the source and target mesh in the same figure.
        """
        pl = pv.Plotter(shape=(1, 3))

        # Plot the source mesh.
        pl.subplot(0, 0)
        pl.add_mesh(source_mesh, color="lightblue")
        pl.add_text(
            f"Source ({source_mesh.n_points} points)",
            font_size=12,
            position="upper_right",
        )

        # Plot the target mesh.
        pl.subplot(0, 1)
        pl.add_mesh(target_mesh, color="orange")
        pl.add_text(
            f"Target ({target_mesh.n_points} points)",
            font_size=12,
            position="upper_right",
        )

        # Plot the source and the target mesh.
        pl.subplot(0, 2)
        pl.add_mesh(source_mesh, color="lightblue", opacity=0.5)
        pl.add_mesh(target_mesh, color="orange", opacity=0.5)
        pl.add_text(
            "Source and target",
            font_size=12,
            position="upper_right",
        )

        pl.show()

    @staticmethod
    def plot_deformed_source(
        source_mesh: pv.PolyData,
        target_mesh: pv.PolyData,
        deformation: np.ndarray,
        show_edges: bool = False,
    ):
        """
        Plot deformed source
        ====================

        Produces a plot of the deformed source and the target.
        """
        # Create a figure with 2 subplots.
        pl = pv.Plotter(shape=(1, 2))

        # Plot the deformed source in the left subplot.
        pl.subplot(0, 0)
        deformed_source_mesh = source_mesh.copy()
        deformed_source_mesh.points = source_mesh.points + deformation
        pl.add_mesh(deformed_source_mesh, color="lightblue", show_edges=show_edges)
        pl.add_text("Deformed source", position="upper_right", font_size=15)

        # Plot the target in the right subplot.
        pl.subplot(0, 1)
        pl.add_mesh(target_mesh, color="orange", show_edges=show_edges)
        pl.add_text("Target", position="upper_right", font_size=15)

        pl.show()

    @staticmethod
    def animate_deformation(
        source_mesh: pv.PolyData,
        target_mesh: pv.PolyData,
        deformation: np.ndarray,
        num_frames: int = 120,
        camera_radius: float = 5,
        camera_elevation: float = 30,
        camera_spin_rate: float = 2,
        show_edges: bool = False,
        show_target: bool = False,
        save_as_gif: bool = False,
        save_as_mp4: bool = False,
    ):
        """
        Animate deformation
        ===================

        Animates the deformation of the source.
        """
        pl = pv.Plotter(off_screen=True)

        # Set the initial camera position.
        pl.camera_position = [
            (camera_radius, 0, 0),
            (0, 0, 0),
            (0, 0, 1),
        ]
        pl.camera.elevation = camera_elevation

        # Generate the frames.
        frames = []
        for frame, t in enumerate(np.linspace(0, 1, num_frames)):
            pl.clear_actors()

            # Compute the deformed source.
            deformed_source = source_mesh.copy()
            deformed_source.points = source_mesh.points + deformation * t
            deformed_source = deformed_source.compute_normals()

            # Plot the deformed source and target (optional).
            if show_target:
                pl.add_mesh(deformed_source, color="lightblue", opacity=0.5)
                pl.add_mesh(target_mesh, color="orange", opacity=0.5)
            else:
                pl.add_mesh(deformed_source, color="lightblue", show_edges=show_edges)

            # Rotate the camera.
            pl.camera.azimuth = frame * camera_spin_rate

            # Save the frame.
            img = pl.screenshot(return_img=True, transparent_background=False)
            frames.append(img)
        pl.close()

        # Create the matplotlib animation.
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis("off")

        # Define how the frames are animated.
        def animate_frame(i):
            ax.clear()
            ax.axis("off")
            ax.imshow(frames[i])
            ax.set_title(f"{i/(len(frames)-1):.1%} deformed", fontsize=16)

        # Create the animation.
        anim = animation.FuncAnimation(
            fig,
            animate_frame,
            frames=len(frames),
            interval=33,
            repeat=True,
            blit=False,
        )
        plt.close()

        # Save the animation.
        if save_as_gif:
            anim.save("deformation_animation.gif", writer="pillow", fps=30)
        if save_as_mp4:
            anim.save("deformation_animation.mp4", writer="ffmpeg", fps=30)

        return HTML(anim.to_jshtml())
