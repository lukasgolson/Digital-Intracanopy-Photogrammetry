import logging

import numpy as np

from numba import njit, prange
from numba.typed import List
from scripts.analysis.processing.hullwrapper import HullWrapper

logger = logging.getLogger(__name__)


def generate_rays(
        shape: tuple[int, int, int],
        num_rays: int,
        z_exclusion: int = 0,
        random_heading: bool = True
) -> np.ndarray:
    """
    Generate ray origin and direction vectors originating from a sphere surrounding a voxel grid,
    facing towards a random position in the voxel grid.

    :param shape: The shape of the interior cube.
    :param num_rays: Number of rays to generate.
    :param z_exclusion: Exclusion angle in degrees from the poles of the sphere.
    :param random_heading: If True, the rays will be assigned a random z heading within the grid.
    :return: A list of tuples (origin, direction) representing rays.
    """
    return np.array(
        _generate_rays(shape, num_rays, z_exclusion, random_heading)
    )


@njit(parallel=True)
def fibonacci_sphere(samples, pole_exclusion_angle=0):
    """
    Generate points on a sphere using the Fibonacci spiral method.

    :param samples: Number of points to generate.
    :param pole_exclusion_angle: The angle in degrees to exclude from the poles of the sphere.
    :return: An array of shape (samples, 3) containing the points on the sphere.
    """
    if samples < 1:
        raise ValueError("At least 1 point is required.")

    points = np.zeros((samples, 3), dtype=np.float64)

    golden_ratio = (1 + np.sqrt(5)) / 2
    pi = np.pi
    # Compute the golden angle (in radians)
    golden_angle = 2 * pi * (1 - 1 / golden_ratio)  # Approximately 2.399963

    # Determine the z limits based on pole exclusion
    if pole_exclusion_angle > 0:
        z_min = np.cos(np.deg2rad(pole_exclusion_angle))
        z_max = -z_min
    else:
        z_min = 1.0
        z_max = -1.0

    for i in prange(samples):
        z = z_min + (z_max - z_min) * (i + 0.5) / samples
        radius = np.sqrt(1 - z * z)
        theta = golden_angle * i

        x = np.cos(theta) * radius
        y = np.sin(theta) * radius

        points[i, 0] = x
        points[i, 1] = y
        points[i, 2] = z

    return points

@njit
def _generate_rays(grid_shape, num_rays, pole_exclusion_angle, random_heading):
    x, y, z = grid_shape
    center = np.array([x / 2, y / 2, z / 2])

    max_value = max(x, y, z)
    radius = np.sqrt(3) * max_value / 2

    sphere = fibonacci_sphere(num_rays, pole_exclusion_angle)

    origins = center + radius * sphere
    directions = List()

    for i in range(num_rays):
        if random_heading:
            target = np.array([np.random.uniform(0, x), np.random.uniform(0, y), np.random.uniform(0, z)])
        else:
            origin = origins[i]
            normal_vector = origin - center
            target = origin - 2 * normal_vector

        direction = target - origins[i]
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Zero-length direction vector encountered.")
        direction = direction / norm
        directions.append(direction)

    return list(zip(origins, directions))


@njit
def within_bounds(point, shape):
    """
    Check if the point is within the bounds of the provided shape.

    :param point: The current point in the grid.
    :param shape: The shape (dimensions) of the grid.
    :return: True if within bounds, False otherwise.
    """

    x, y, z = point
    max_x = shape[0]
    max_y = shape[1]
    max_z = shape[2]
    return 0 <= x < max_x and 0 <= y < max_y and 0 <= z < max_z


@njit
def march_ray(grid, equations, origin, direction, step_size=0.5, max_steps=3000):
    photon = origin
    direction = direction / np.linalg.norm(direction)
    hits = List()
    passed_hull = False

    while max_steps > 0:
        max_steps -= 1
        photon += direction * step_size

        # Check if the photon has passed through the hull, regardless of grid bounds
        if not passed_hull and points_in_hull(photon, equations):
            passed_hull = True

        # Now check if the photon is within the grid bounds
        if not within_bounds(photon, grid.shape):
            # If the photon is outside the grid and has already passed the hull, we can stop
            if passed_hull:
                break  # Exit the loop if we've passed the hull and are now outside the grid
            else:
                continue  # Continue iterating if we haven't passed the hull yet

        # Proceed with the rest of the logic
        voxel_pos = np.array([int(photon[0]), int(photon[1]), int(photon[2])], dtype=np.int32)
        voxel = grid[voxel_pos[0], voxel_pos[1], voxel_pos[2]]

        if voxel[4] > 0.0:
            hits.append(voxel_pos)

    return hits, passed_hull


@njit
def points_in_hull(point, hull_equations, tolerance=1e-12):

    if len(hull_equations) == 0:
        return False

    for eq in hull_equations:
        if np.dot(eq[:-1], point) + eq[-1] > tolerance:
            return False
    return True


def process_rays(voxel_grid: np.ndarray, rays: np.ndarray, hull: HullWrapper, min_hit_requirement: int = 2,
        initial_ray_energy: float = 1, energy_threshold: float = 0.5, ray_dissipation_factor: float = 0.5, step_size: float = 0.5) -> tuple[float, int]:

    hull_equations = hull.get_equations()
    if hull_equations is None:
        return 0, 0

    ray_results = step_rays(voxel_grid, hull_equations, rays, step_size)

    transmitted = calculate_transmission(
        voxel_grid.shape[:3],
        ray_results,
        initial_ray_energy,
        min_hit_requirement,
        energy_threshold,
        ray_dissipation_factor
    )

    return transmitted, len(ray_results)


@njit()
def step_rays(grid, hull_equations, rays, step_size = 0.5):

    results = List()

    for ray in rays:
        start_pos = ray[0]
        direction = ray[1]

        hits, passed_hull = march_ray(grid, hull_equations, start_pos, direction, step_size)

        if passed_hull:
            results.append(hits)

    return results


@njit(parallel=True)
def calculate_transmission(
        grid_shape,
        ray_results,
        initial_ray_energy,
        minimum_voxel_hit_threshold,
        energy_threshold=0.25,
        ray_dissipation_factor=0.5,
):

    dissipation_factor = 1 - ray_dissipation_factor

    voxel_hits = np.zeros(grid_shape, dtype=np.float64)
    for ray in ray_results:
        hits = ray
        for hit in hits:
            x, y, z = hit
            voxel_hits[x, y, z] += 1

    unobstructed_array = np.zeros(len(ray_results), dtype=np.int32)
    for i in prange(len(ray_results)):
        hits = ray_results[i]

        energy = initial_ray_energy

        for hit in hits:

            if voxel_hits[hit[0], hit[1], hit[2]] >= minimum_voxel_hit_threshold:
                energy = (energy * dissipation_factor)

        if energy > energy_threshold:
            unobstructed_array[i] = 1

    unobstructed_ray_count = unobstructed_array.sum()
    return unobstructed_ray_count


# a helper function to save the rays to a text file
def save_rays(rays, file: str = "rays.txt", x_offset=0, y_offset=0, z_offset=0):
    """
    Save ray origins with their corresponding directions represented as colors to a text file.

    :param rays: Array of rays where each ray is represented as (origin, direction).
    :param file: The filename to save the rays.
    :param x_offset: Offset to apply to the x-coordinate of the ray origins.
    :param y_offset: Offset to apply to the y-coordinate of the ray origins.
    :param z_offset: Offset to apply to the z-coordinate of the ray origins.
    """
    origins = rays[:, 0]  # Get origin vectors
    directions = rays[:, 1]  # Get direction vectors

    adjusted_origins = origins + np.array([x_offset, y_offset, z_offset])

    normalized_directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    color_directions = ((normalized_directions + 1) / 2 * 255).astype(np.uint8)

    combined_data = np.hstack((adjusted_origins, color_directions))

    np.savetxt(file, combined_data, delimiter=' ', fmt="%.6f %.6f %.6f %d %d %d")

    logger.info(f"Rays saved to {file}.")
