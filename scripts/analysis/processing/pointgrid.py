import logging

import numpy as np

from numba import njit, prange

logger = logging.getLogger(__name__)





def adjust_classification(classification):
    unique_values, inverse = np.unique(classification, return_inverse=True)
    adjusted_classification = inverse.reshape(classification.shape)
    return adjusted_classification


@njit(parallel=True)
def normalize_grid(grid: np.ndarray, noise_floor):
    normalized_grid = np.zeros(grid.shape[:-1] + (7,)).astype(np.float32)
    # Calculate grid density
    high_density_count = 0
    total_count = 0
    # Loop through the voxel grid and accumulate the sums based on density values

    epsilon = np.finfo(np.float32).eps

    for x in prange(grid.shape[0]):
        for y in prange(grid.shape[1]):
            for z in prange(grid.shape[2]):
                voxel = grid[x, y, z]

                red = voxel[0]
                green = voxel[1]
                blue = voxel[2]
                density = voxel[3]

                intensity = red + green + blue

                if intensity != 0:
                    red_norm = red / intensity
                    green_norm = green / intensity
                    blue_norm = blue / intensity
                else:
                    red_norm = 0
                    green_norm = 0
                    blue_norm = 0

                rgri = (red - green) / (red + green + epsilon)
                bgri = (blue - green) / (blue + green + epsilon)

                luminosity = (red + green + blue) / 3

                max_proportion = max(red_norm, green_norm, blue_norm)
                min_proportion = min(red_norm, green_norm, blue_norm)

                color_balance_score = 1 - (max_proportion - min_proportion)

                if density > 0:
                    total_count += 1
                if density > noise_floor:
                    high_density_count += 1

                # Store the calculated values in the new_array at the same index position as voxel
                normalized_grid[x, y, z] = [
                    red_norm,
                    green_norm,
                    blue_norm,
                    rgri,
                    bgri,
                    luminosity,
                    color_balance_score,
                ]
    grid_density = (high_density_count / total_count) * 100

    return normalized_grid, grid_density



def accumulate_points(
    point_cloud: np.ndarray,
    voxel_size: float = 0.1,
    filter_by_percentile: bool = True,
    outlier_percentage: float = 5.0,
):
    """
    Accumulate points into a voxel grid.
    :param outlier_percentage:
    :param filter_by_percentile:
    :param point_cloud: A 2D NumPy array with shape (N, 8) where N is the number of points in the point cloud.
    :param voxel_size: The size of each voxel in meters.
    :return: A 4D NumPy array with shape (X, Y, Z, 6) where X, Y, and Z are the dimensions of the voxel grid.
    index 0 = normalized red; index 1 = normalized green; index 2 = normalized blue
    index 3 = relative density; index 4 = classification; index 5 = tree id
    """
    # Calculate the minimum and maximum bounds for the 3D point cloud
    min_bound = np.min(point_cloud[:, :3], axis=0)
    max_bound = np.max(point_cloud[:, :3], axis=0)
    grid_side_length = max_bound - min_bound
    grid_dimensions = np.ceil(grid_side_length / voxel_size).astype(np.int64) + 1

    if filter_by_percentile:
        voxel_grid, color_grid, class_grid = accumulate_points_with_percentile_filter(
            point_cloud,
            voxel_size,
            min_bound,
            grid_dimensions,
            lower_percentile=outlier_percentage,
            upper_percentile=100 - outlier_percentage,
        )
    else:
        voxel_grid, color_grid, class_grid = numba_accumulate_points(
            point_cloud, voxel_size, min_bound, grid_dimensions
        )

        # Calculate the minimum and maximum bounds for the 3D point cloud
    combined = combine_grids(voxel_grid, color_grid, class_grid)
    logger.info("Combined grids")
    return combined


def accumulate_points_with_percentile_filter(
    pointcloud: np.ndarray,
    voxel_size: float,
    min_bound: np.ndarray,
    grid_dimensions: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
):
    inv_voxel_size = 1.0 / voxel_size

    voxel_indices = ((pointcloud[:, :3] - min_bound) * inv_voxel_size).astype(np.int64)

    flat_voxel_indices = np.ravel_multi_index(
        (voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]),
        dims=grid_dimensions,
    )

    voxel_grid = np.zeros(
        (grid_dimensions[0], grid_dimensions[1], grid_dimensions[2], 1), dtype=np.int64
    )
    color_grid = np.zeros(
        (grid_dimensions[0], grid_dimensions[1], grid_dimensions[2], 3),
        dtype=np.float64,
    )
    classification_grid = np.zeros(
        (grid_dimensions[0], grid_dimensions[1], grid_dimensions[2], 3), dtype=np.int64
    )

    unique_voxels, inverse_indices, counts = np.unique(
        flat_voxel_indices, return_inverse=True, return_counts=True
    )

    progress_5 = int(len(unique_voxels) * 0.05)

    for i, flat_idx in enumerate(unique_voxels):
        indices = np.where(inverse_indices == i)[0]
        if indices.size == 0:
            continue

        if i % progress_5 == 0:
            logger.info(f"Processing voxel {i} of {unique_voxels.size}")

        colors = pointcloud[indices, 3:6]

        lower_thresh = np.percentile(colors, lower_percentile, axis=0)
        upper_thresh = np.percentile(colors, upper_percentile, axis=0)

        mask = np.all((colors >= lower_thresh) & (colors <= upper_thresh), axis=1)
        filtered_indices = indices[mask]

        if filtered_indices.size == 0:
            continue

        voxel_coord = np.unravel_index(flat_idx, grid_dimensions)

        voxel_grid[voxel_coord[0], voxel_coord[1], voxel_coord[2], 0] = (
            filtered_indices.size
        )

        color_sum = np.sum(pointcloud[filtered_indices, 3:6].astype(np.float64), axis=0)
        color_grid[voxel_coord[0], voxel_coord[1], voxel_coord[2], :] = color_sum


        ground_count = np.sum(pointcloud[filtered_indices, 6] == 1)
        non_ground_count = filtered_indices.size - ground_count
        classification_grid[voxel_coord[0], voxel_coord[1], voxel_coord[2], 0] = (
            ground_count
        )
        classification_grid[voxel_coord[0], voxel_coord[1], voxel_coord[2], 1] = (
            non_ground_count
        )

        classification_grid[voxel_coord[0], voxel_coord[1], voxel_coord[2], 2] = (
            pointcloud[filtered_indices[0], 7]
        )

    return voxel_grid, color_grid, classification_grid


@njit(parallel=True, fastmath=True)
def numba_accumulate_points(
    pointcloud: np.ndarray,
    voxel_size: float,
    min_bound: np.ndarray,
    grid_dimensions: np.ndarray,
):

    # doesn't noticeably impact Numba performance, but appears to help with pure Python performance
    inv_voxel_size = 1.0 / voxel_size

    # Create an empty voxel grid; each cell has 5 channels.
    # 0 is point count;
    voxel_grid = np.zeros(
        (grid_dimensions[0], grid_dimensions[1], grid_dimensions[2], 1), dtype=np.int64
    )

    # 0,1,2 is RGB info;
    color_grid = np.zeros(
        (grid_dimensions[0], grid_dimensions[1], grid_dimensions[2], 3),
        dtype=np.float64,
    )

    # 0 is classification; e.g., ground or non-ground
    classification_grid = np.zeros(
        (grid_dimensions[0], grid_dimensions[1], grid_dimensions[2], 3), dtype=np.int64
    )

    for point_index in prange(pointcloud.shape[0]):
        point = pointcloud[point_index]
        # Calculate the voxel index for each point in the voxel grid based on the point's position
        # voxel_index = ((point[:3] - min_bound) / voxel_size).astype(int)
        voxel_index = ((point[:3] - min_bound) * inv_voxel_size).astype(np.int64)

        # Update voxel point count
        voxel_grid[voxel_index[0], voxel_index[1], voxel_index[2], 0] += 1  # Count

        # Extract the voxel color (R, G, B) from the point's last three elements and accumulate it
        color_grid[voxel_index[0], voxel_index[1], voxel_index[2], :3] += point[
            3:6
        ].astype(
            np.float64
        )  # Accumulate color

        # Update voxel classification
        # if the point is classified as ground, set the first channel to 1; second channel to 0
        if point[6] == 1:
            classification_grid[voxel_index[0], voxel_index[1], voxel_index[2], 0] += 1
        else:
            classification_grid[voxel_index[0], voxel_index[1], voxel_index[2], 1] += 1

        classification_grid[voxel_index[0], voxel_index[1], voxel_index[2], 2] = point[
            7
        ]

    return voxel_grid, color_grid, classification_grid


def combine_grids(point_grid, color_grid, classification_grid, percentile=100):
    logger.info("Creating voxel mask...")
    nonzero_count_mask = point_grid[..., 0] != 0
    nonzero_counts = point_grid[nonzero_count_mask, 0]
    nonzero_total = np.where(nonzero_count_mask, point_grid[..., 0], 0.01)

    percentile_of_points = np.percentile(nonzero_counts, percentile)
    logger.info("Normalizing voxel color")
    final_grid = np.zeros(
        (point_grid.shape[0], point_grid.shape[1], point_grid.shape[2], 6),
        dtype=np.float32,
    )
    final_grid[..., 0:3] = (
        color_grid[..., 0:3] * (1.0 / nonzero_total[..., np.newaxis])
    ) / 255.0

    final_grid[..., 3] = np.minimum(point_grid[..., 0] / percentile_of_points, 1.0)

    # calculate the classification of each voxel;
    # if the voxel is empty, set the classification to 0
    # if the voxel is non-empty:
    # 1. if the first channel of classification_grid is greater than the second channel, set the classification to 0
    # 2. if the second channel of classification_grid is greater than the first channel, set the classification to 1

    final_grid[..., 4] = np.where(
        nonzero_count_mask,
        np.where(classification_grid[..., 0] > classification_grid[..., 1], 2, 4),
        0,
    )

    final_grid[..., 5] = classification_grid[..., 2]

    return final_grid
