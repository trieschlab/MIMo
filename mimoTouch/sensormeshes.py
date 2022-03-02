import math

import numpy as np
from scipy.spatial import Delaunay
import trimesh.points
import trimesh.util

from mimoEnv.utils import EPS
import mimoTouch.sensorpoints


def mesh_box(resolution: float, sizes: np.array):
    """Spreads points over a box by subdividing into smaller boxes.
    Sizes is an array containing the three edge half-lengths of the box"""
    assert len(sizes) == 3, "Size parameter does not fit box!"
    # If distance between points is greater than size of box, have only one point in center of box
    if resolution > sizes.max():
        return trimesh.points.PointCloud(np.zeros((1, 3), dtype=np.float64))

    # Otherwise subdivide box into smaller boxes
    n_divisions = np.maximum(2, np.ceil(2*sizes / resolution).astype(np.int64))
    # Fill points with actual sensor positions
    x_coords = np.linspace(-sizes[0], sizes[0], n_divisions[0])
    y_coords = np.linspace(-sizes[1], sizes[1], n_divisions[1])
    z_coords = np.linspace(-sizes[2], sizes[2], n_divisions[2])

    faces = []
    points = []
    # Generate all the vertices. This will duplicate vertices at the edges/corners but we will fix that later
    points.extend([(x_coords[0], y, z) for y in y_coords for z in z_coords])
    points.extend([(x_coords[-1], y, z) for y in y_coords for z in z_coords])
    for y in range(1, n_divisions[1]):
        for z in range(n_divisions[2]):
            cur_idx = y * n_divisions[2] + z
            cur_idx1 = cur_idx + (n_divisions[1]*n_divisions[2])
            if z > 0:
                faces.append([cur_idx, cur_idx - 1, cur_idx - 1 - n_divisions[2]])
                faces.append([cur_idx, cur_idx - 1 - n_divisions[2], cur_idx - n_divisions[2]])
                faces.append([cur_idx1, cur_idx1 - 1, cur_idx1 - 1 - n_divisions[2]])
                faces.append([cur_idx1, cur_idx1 - 1 - n_divisions[2], cur_idx1 - n_divisions[2]])

    point_offset = len(points)
    points.extend([(x, y_coords[0], z) for x in x_coords for z in z_coords])
    points.extend([(x, y_coords[-1], z) for x in x_coords for z in z_coords])
    for x in range(1, n_divisions[0]):
        for z in range(n_divisions[2]):
            cur_idx = x * n_divisions[2] + z + point_offset
            cur_idx1 = cur_idx + (n_divisions[0]*n_divisions[2])
            if z > 0:
                faces.append([cur_idx, cur_idx - 1, cur_idx - 1 - n_divisions[2]])
                faces.append([cur_idx, cur_idx - 1 - n_divisions[2], cur_idx - n_divisions[2]])
                faces.append([cur_idx1, cur_idx1 - 1, cur_idx1 - 1 - n_divisions[2]])
                faces.append([cur_idx1, cur_idx1 - 1 - n_divisions[2], cur_idx1 - n_divisions[2]])

    point_offset = len(points)
    points.extend([(x, y, z_coords[0]) for x in x_coords for y in y_coords])
    points.extend([(x, y, z_coords[-1]) for x in x_coords for y in y_coords])
    for x in range(1, n_divisions[0]):
        for y in range(n_divisions[1]):
            cur_idx = x * n_divisions[1] + y + point_offset
            cur_idx1 = cur_idx + (n_divisions[0]*n_divisions[1])
            if y > 0:
                faces.append([cur_idx, cur_idx - 1, cur_idx - 1 - n_divisions[1]])
                faces.append([cur_idx, cur_idx - 1 - n_divisions[1], cur_idx - n_divisions[1]])
                faces.append([cur_idx1, cur_idx1 - 1, cur_idx1 - 1 - n_divisions[1]])
                faces.append([cur_idx1, cur_idx1 - 1 - n_divisions[1], cur_idx1 - n_divisions[1]])

    mesh = trimesh.Trimesh(vertices=np.asarray(points), faces=np.asarray(faces))
    mesh.merge_vertices()  # Merge vertices to turn our 6 disconnected surfaces into a single mesh
    mesh.fix_normals()

    return mesh


def mesh_sphere(resolution: float, radius: float):
    # If resolution would lead to very small number of sensor points, instead have single point at center of sphere
    points = mimoTouch.sensorpoints.spread_points_sphere(resolution=resolution, radius=radius)
    mesh = trimesh.points.PointCloud(points)
    if points.shape[0] == 1:
        return mesh
    else:
        return mesh.convex_hull


def mesh_ellipsoid(resolution: float, radii):
    points = mimoTouch.sensorpoints.spread_points_ellipsoid(resolution=resolution, radii=radii)
    mesh = trimesh.points.PointCloud(points)
    if points.shape[0] == 1:
        return mesh
    else:
        return mesh.convex_hull


def mesh_pipe(resolution: float, length: float, radius: float):
    """ Spreads points around the outer surface of a cylinder, without caps. """
    # Number of subdivisions along length
    n_length = int(math.ceil(length / resolution))
    # Number of subdivisions around circumference
    n_circum = int(math.ceil(2 * math.pi * radius / resolution))

    points = []
    faces = []
    for i in range(n_length):
        if n_length == 1:
            z = 0
        else:
            z = (i * length / (n_length - 1)) - length / 2
        for j in range(n_circum):
            theta = 2 * math.pi * j / n_circum
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            points.append([x, y, z])
            cur_idx = i * n_circum + j
            if i > 0 and j > 0:
                faces.append([cur_idx, cur_idx - 1, cur_idx - 1 - n_circum])
                faces.append([cur_idx, cur_idx - 1 - n_circum, cur_idx - n_circum])
            elif i > 0 and j == 0:
                faces.append([cur_idx, cur_idx + n_circum - 1, cur_idx - 1])
                faces.append([cur_idx, cur_idx - 1, cur_idx - n_circum])

    mesh = trimesh.Trimesh(vertices=np.asarray(points), faces=np.asarray(faces))
    mesh.fix_normals()
    return mesh


def mesh_cylinder(resolution: float, length: float, radius: float):
    """ Spreads points around the outer surface of a cylinder, including caps """
    # Number of subdivisions along length
    n_length = int(math.ceil(length / resolution))
    # Number of subdivisions around circumference
    n_circum = int(math.ceil(2 * math.pi * radius / resolution))
    # Number of rings on caps
    n_caps = int(round(radius / resolution)) + 1

    # If resolution is too low to cover cylinder at least roughly, return single point centered on cylinder
    if n_circum < 3 or n_length < 2 or n_caps == 1:
        return trimesh.points.PointCloud(np.zeros((1, 3), dtype=np.float64))

    pipe_mesh = mesh_pipe(resolution, length, radius)

    # Caps
    cap_points = []
    face_offset = pipe_mesh.vertices.shape[0]

    cap_radii = np.linspace(0, radius, n_caps)
    z = length / 2
    # Go through the radii
    for cap_radius in cap_radii:
        # Calculate the number of subdivisions
        if abs(cap_radius) < EPS:
            n_cap_circum = 1
        else:
            n_cap_circum = int(math.ceil(2 * math.pi * cap_radius / resolution))
        for j in range(n_cap_circum):
            theta = 2 * math.pi * j / n_cap_circum
            x = cap_radius * math.cos(theta)
            y = cap_radius * math.sin(theta)
            cap_points.append([x, y])

    cap_points = np.asarray(cap_points)
    # Do Delaunay of caps
    tri = Delaunay(cap_points, qhull_options="Qbb Qc Q12 Qt")
    # Upper cap
    faces_u = tri.simplices + face_offset
    points_u = np.pad(tri.points.copy(), ((0, 0), (0, 1)), mode="constant", constant_values=z)
    # Lower Cap
    faces_l = tri.simplices + tri.points.shape[0] + face_offset
    points_l = np.pad(tri.points.copy(), ((0, 0), (0, 1)), mode="constant", constant_values=-z)
    # Merge everything
    points = np.concatenate([pipe_mesh.vertices, points_u, points_l])
    faces = np.concatenate([pipe_mesh.faces, faces_u, faces_l])
    mesh = trimesh.Trimesh(vertices=points, faces=faces)
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def mesh_capsule(resolution: float, length: float, radius: float):
    # Number of subdivisions around circumference
    n_circum = int(math.ceil(2 * math.pi * radius / resolution))

    # If resolution is too low to cover geom at least roughly, return single point centered on cylinder
    if n_circum < 3:
        return trimesh.points.PointCloud(np.zeros((1, 3), dtype=np.float64))

    pipe_mesh = mesh_pipe(resolution, length, radius)

    # Make two half-spheres using our points and trimeshes convex hull
    sphere_points_u = []
    sphere_points_l = []
    n_theta = int(math.ceil(math.pi * radius / (2*resolution)))
    for i in range(n_theta + 1):
        theta = (i + 0) * math.pi / (2*n_theta)
        n_phi = math.ceil(2 * math.pi * math.sin(theta) * radius / resolution)
        if n_phi < 2:
            z = length / 2 + radius
            sphere_points_u.append([0, 0, z])
            sphere_points_l.append([0, 0, -z])
        else:
            for j in range(n_phi):
                phi = 2 * math.pi * j / n_phi
                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z = radius * math.cos(theta) + length / 2
                sphere_points_u.append([x, y, z])
                sphere_points_l.append([x, y, -z])

    upper_hemi = trimesh.points.PointCloud(np.asarray(sphere_points_u)).convex_hull
    lower_hemi = trimesh.points.PointCloud(np.asarray(sphere_points_l)).convex_hull
    # Filter out internal faces
    upper_faces = upper_hemi.faces[np.dot(upper_hemi.face_normals, np.array([0, 0, 1])) > -EPS]
    lower_faces = lower_hemi.faces[np.dot(lower_hemi.face_normals, np.array([0, 0, -1])) > -EPS]

    face_offset = pipe_mesh.vertices.shape[0]
    points = np.concatenate([pipe_mesh.vertices, upper_hemi.vertices, lower_hemi.vertices])
    faces = np.concatenate([pipe_mesh.faces,
                            upper_faces + face_offset,
                            lower_faces + face_offset + upper_hemi.vertices.shape[0]])
    mesh = trimesh.Trimesh(vertices=points, faces=faces)
    mesh.merge_vertices()
    mesh.fix_normals()

    return mesh
