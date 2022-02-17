import math

import numpy as np

from mimoEnv.utils import EPS


# TODO: function that spreads points over arbitrary mesh using spherical/cubic projection.
#       Take mujoco mesh and do intersections there?
#       Maybe mujoco raycast methods can be used here, if we can exclude other objects?

def spread_points_box(resolution: float, sizes: np.array):
    """Spreads points over a box by subdividing into smaller boxes.
    Sizes is an array containing the three edge lengths of the box"""
    assert len(sizes) == 3, "Size parameter does not fit box!"
    # If distance between points is greater than size of box, have only one point in center of box
    if resolution > sizes.max():
        return np.zeros((1, 3), dtype=np.float64)

    # Otherwise subdivide box into smaller boxes
    n_divisions = np.maximum(2, np.ceil(2*sizes / resolution).astype(np.int64))
    # Fill points with actual sensor positions
    x_coords = np.linspace(-sizes[0], sizes[0], n_divisions[0])
    y_coords = np.linspace(-sizes[1], sizes[1], n_divisions[1])
    z_coords = np.linspace(-sizes[2], sizes[2], n_divisions[2])

    # All points were one axis is either +1 or -1
    x0_points = [(x_coords[0], y, z) for y in y_coords for z in z_coords]
    x1_points = [(x_coords[-1], y, z) for y in y_coords for z in z_coords]
    y0_points = [(x, y_coords[0], z) for x in x_coords[1:-1] for z in z_coords]
    y1_points = [(x, y_coords[-1], z) for x in x_coords[1:-1] for z in z_coords]
    z0_points = [(x, y, z_coords[0]) for x in x_coords[1:-1] for y in y_coords[1:-1]]
    z1_points = [(x, y, z_coords[-1]) for x in x_coords[1:-1] for y in y_coords[1:-1]]

    points = x0_points + x1_points + y0_points + y1_points + z0_points + z1_points
    points = np.asarray(points)
    return points


def spread_points_sphere(resolution: float, radius: float):
    # If resolution would lead to very small number of sensor points, instead have single point at center of sphere
    if resolution > radius:
        return np.zeros((1, 3), dtype=np.float64)

    points = []

    n_theta = int(math.ceil(math.pi * radius / resolution))

    for i in range(n_theta):
        theta = (i + 0.5) * math.pi / n_theta
        n_phi = round(2 * math.pi * math.sin(theta) * radius / resolution)
        for j in range(n_phi):
            phi = 2*math.pi*j / n_phi
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            points.append([x, y, z])

    return np.array(points)


# TODO: Proper points instead of spherical projection
def spread_points_ellipsoid(resolution: float, radii):
    """ Spread points over an ellipsoid using a spherical projection. radii should be list/array (a, b, c), where
    a is the x-radius, b the y-radius and c the z-radius"""
    max_r = np.max(radii)
    # If resolution would lead to very small number of sensor points, instead have single point at center of sphere
    if resolution > max_r:
        return np.zeros((1, 3), dtype=np.float64)

    points = []

    n_theta = int(math.ceil(math.pi * max_r / resolution))

    for i in range(n_theta):
        theta = (i + 0.5) * math.pi / n_theta
        n_phi = round(2 * math.pi * math.sin(theta) * max_r / resolution)
        for j in range(n_phi):
            phi = 2*math.pi*j / n_phi
            x = radii[0] * math.sin(theta) * math.cos(phi)
            y = radii[1] * math.sin(theta) * math.sin(phi)
            z = radii[2] * math.cos(theta)
            points.append([x, y, z])

    return np.array(points)


def spread_points_pipe(resolution: float, length: float, radius: float):
    """ Spreads points around the outer surface of a cylinder, without caps """
    # Number of subdivisions along length
    n_length = int(math.ceil(length / resolution))
    # Number of subdivisions around circumference
    n_circum = int(math.ceil(2 * math.pi * radius / resolution))

    points = []
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
    return np.array(points)


def spread_points_cylinder(resolution: float, length: float, radius: float):
    """ Spreads points around the outer surface of a cylinder, including caps """
    # Number of subdivisions along length
    n_length = int(math.ceil(length / resolution))
    # Number of subdivisions around circumference
    n_circum = int(math.ceil(2 * math.pi * radius / resolution))
    # Number of rings on caps
    n_caps = int(round(radius / resolution)) + 1

    # If resolution is too low to cover cylinder at least roughly, return single point centered on cylinder
    if n_circum < 3 or n_length < 2:
        return np.zeros((1, 3), dtype=np.float64)

    pipe_points = spread_points_pipe(resolution, length, radius)

    # Caps
    cap_points = []
    if n_caps > 1:
        cap_radii = np.linspace(0, radius, n_caps)
        # Go through the radii, except for the outermost, which we already covered
        for cap_radius in cap_radii[:-1]:
            # Calculate the number of subdivisions
            if abs(cap_radius) < EPS:
                n_cap_circum = 1
            else:
                n_cap_circum = int(math.floor(2 * math.pi * cap_radius / resolution))
            for j in range(n_cap_circum):
                theta = 2 * math.pi * (j + 0.5) / n_cap_circum
                x = cap_radius * math.cos(theta)
                y = cap_radius * math.sin(theta)
                cap_points.append([x, y, -length / 2])
                cap_points.append([x, y, length / 2])
    if len(cap_points) == 0:
        return pipe_points
    else:
        return np.concatenate([pipe_points, np.array(cap_points)])


def spread_points_capsule(resolution: float, length: float, radius: float):
    # Number of subdivisions around circumference
    n_circum = int(math.ceil(2 * math.pi * radius / resolution))

    # If resolution is too low to cover geom at least roughly, return single point centered on cylinder
    if n_circum < 3:
        return np.zeros((1, 3), dtype=np.float64)

    pipe_points = spread_points_pipe(resolution, length, radius)

    # Two half-spheres with open edge already covered
    sphere_points = []
    n_theta = int(round(math.pi * radius / (2*resolution)))
    for i in range(n_theta):
        theta = (i + 0) * math.pi / (2*n_theta)
        n_phi = round(2 * math.pi * math.sin(theta) * radius / resolution)
        if n_phi < 2:
            z = length / 2 + radius
            sphere_points.append([0, 0, z])
            sphere_points.append([0, 0, -z])
        else:
            for j in range(n_phi):
                phi = 2 * math.pi * j / n_phi
                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z = radius * math.cos(theta) + length / 2  # half spheres at end of cylinder
                sphere_points.append([x, y, z])
                sphere_points.append([x, y, -z])

    return np.concatenate([pipe_points, np.array(sphere_points)])
