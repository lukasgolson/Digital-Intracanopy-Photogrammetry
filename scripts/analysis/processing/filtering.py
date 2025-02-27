import numpy as np
import logging

import scipy
from scipy.ndimage import distance_transform_edt, convolve

logger = logging.getLogger(__name__)

NOISE_THRESHOLD = 0.25
HIGH_PASS_THRESHOLD = 0.1


def apply_filters_and_masks(combined_grid, density_threshold=0.25):
    """
    Applies a series of filters and masks to the combined grid to remove noise and isolate the main tree.
    :param combined_grid: The combined grid containing the data.
    :param density_threshold:
    """
    filter = [remove_ground]

    masks = [
        lambda x: create_density_mask(x, density_threshold),
    ]

    logger.info("Applying filters and masks to the combined grid.")

    post_filter = [
        #     lambda x: denoise_and_grow_main_tree(
        #         x, denoise_size_threshold, distance_threshold
        #     ),
        lambda x: trim_grid_to_voxels(x),
    ]

    channel_functions = [
        calculate_gli,
        # remap the density channel;
        lambda grid, new_index: passthrough_channel(
            grid, new_index, 3, normalize_channel
        ),
        calculate_normalized_difference,
        calculate_exr,
    ]


    return filter_enrich_grid(
        combined_grid, filter, masks, post_filter, channel_functions
    )


def trim_grid_to_voxels(grid):
    # Identify the slices for non-zero entries across the first three dimensions of the 4th dimension
    nonzero_slices = np.any(grid[:, :, :, :3] != 0, axis=3)
    # Get the indices of all non-zero entries based on the slices
    nonzero_indices = np.nonzero(nonzero_slices)
    # Get the minimum and maximum indices for each dimension
    min_indices = np.min(nonzero_indices, axis=1)
    max_indices = np.max(nonzero_indices, axis=1)
    # Create a new grid with the minimum size that encapsulates all non-zero entries
    shrunk_grid = grid[
        min_indices[0] : max_indices[0] + 1,
        min_indices[1] : max_indices[1] + 1,
        min_indices[2] : max_indices[2] + 1 :,
    ]

    return shrunk_grid


def create_noise_floor_mask(channels, threshold):
    # Calculate the sum of the RGB channels which are assumed to be the first three channels
    rgb_channels = channels[..., :3]

    sum_rgb = np.sum(rgb_channels, axis=-1)
    return sum_rgb > threshold


def create_density_mask(voxel_grid, threshold=0.25):
    """
    Computes a binary mask from the density channel (index 3) of the voxel grid.
    Voxels with density less than the threshold are marked as False.

    Parameters:
    - voxel_grid (np.ndarray): The 4D voxel grid.
    - threshold (float): The density threshold below which voxels are removed.

    Returns:
    - np.ndarray: A boolean mask with True for voxels with density >= threshold.
    """
    # Extract the density channel (channel index 3)
    density_channel = voxel_grid[:, :, :, 3]

    # Create a mask that is True where density >= threshold
    mask = density_channel >= threshold
    return mask


def remove_ground(combined_grid):
    """
    Filters out the non tree points from the combined grid.

    Parameters:
    - combined_grid (numpy.ndarray): The 4D grid containing the data.

    Returns:
    - numpy.ndarray: The filtered grid with ground points removed.
    """
    # Initialize the filtered grid
    filtered_grid = np.copy(combined_grid)

    filtered_grid[filtered_grid[:, :, :, 4] == 2, :3] = 0
    filtered_grid[filtered_grid[:, :, :, 4] == 0, :3] = 0

    return filtered_grid


def highpass(channels, threshold):
    rgb_channels = channels[..., :3]

    sum_rgb = np.sum(rgb_channels, axis=-1)
    return sum_rgb > threshold


def lowpass(x, threshold):
    mask = x >= threshold
    return mask


def proximity_denoised_mask(channels, threshold):
    # Define a 3x3x3 kernel for averaging neighbors in 3D space
    kernel = np.ones((3, 3, 3)) / 27

    # Compute the average of the first three channels across all neighboring cells
    # This involves averaging across channels first, then convolving in spatial dimensions
    mean_channels = np.mean(channels[..., :3], axis=-1)
    local_density = convolve(mean_channels, kernel, mode="constant", cval=0.0)

    # Create a mask based on the local density threshold
    # True where the local density is greater than the threshold, indicating sufficient proximity of similar values
    mask = local_density > threshold

    return mask


def denoise_and_grow_main_tree(
    voxel_grid, size_threshold: int = 25, distance_threshold: int = 10
):

    logger.info(np.shape(voxel_grid))
    # Step 1: Convert to binary format
    binary_voxel_grid = (voxel_grid[:, :, :, 4] > 0).astype(int)

    logger.info("Performing connected component analysis.")

    # Step 2: Apply connected component analysis
    labeled_grid, num_features = scipy.ndimage.label(binary_voxel_grid)

    # Step 3: Measure the size of each component
    component_sizes = np.bincount(labeled_grid.ravel())

    logger.info("Identifying the largest component.")

    # Identify the largest component (main tree)
    largest_component_label = component_sizes.argmax()

    # Create a binary mask for the largest component
    main_tree_mask = labeled_grid == largest_component_label

    # Remove components smaller than the size threshold
    for i in range(1, num_features + 1):
        if component_sizes[i] < size_threshold:
            main_tree_mask[labeled_grid == i] = 0

    # Step 4: Calculate distance from each voxel to the main tree
    distance_to_main_tree = distance_transform_edt(~main_tree_mask)

    # Create a mask for components that are within the distance threshold
    add_nearby_components_mask = (distance_to_main_tree <= distance_threshold) & (
        labeled_grid > 0
    )

    # Combine the main tree mask with the nearby components mask
    grown_tree_mask = main_tree_mask | add_nearby_components_mask

    # Apply the mask to the original voxel grid
    grown_voxel_grid = np.copy(voxel_grid)
    grown_voxel_grid[~grown_tree_mask] = 0

    return grown_voxel_grid


def filter_enrich_grid(
    voxel_grid: np.ndarray,
    filter_functions=None,
    masks=None,
    post_filters=None,
    channel_functions=None,
):
    if filter_functions is None:
        filter_functions = []

    if masks is None:
        masks = []

    if post_filters is None:
        post_filters = []

    if channel_functions is None:
        channel_functions = []

    for filter_function in filter_functions:
        logger.debug("Applying filter function")
        voxel_grid = filter_function(voxel_grid)

    # Apply each mask in the list to the filtered_grid
    for mask_func in masks:
        logger.debug("Applying mask function")
        mask = mask_func(voxel_grid)
        voxel_grid *= mask[..., np.newaxis]  # Apply mask to each channel

        # print 3rd index to see if it is the density channel

    for post_filter in post_filters:
        logger.debug("Applying post filter function")
        voxel_grid = post_filter(voxel_grid)

    # Start with the original voxel grid reduced to its first three channels (assuming RGB)
    new_grid = voxel_grid[:, :, :, :3]

    # Apply each channel function to the original channels and append the result to the RGB channels
    channel_id = 3  # Start with channel 3
    for func in channel_functions:
        new_channel = func(voxel_grid, channel_id)
        new_grid = np.concatenate((new_grid, new_channel[..., np.newaxis]), axis=-1)
        channel_id += 1

    return new_grid


def calculate_gli(channels, index):
    red, green, blue = channels[..., 0], channels[..., 1], channels[..., 2]

    # Calculate the denominator
    denominator = (2 * green) + red + blue

    # Initialize the GLI array with zeros
    gli = np.zeros_like(denominator, dtype=float)

    # Find non-zero denominators to avoid division by zero
    valid = denominator != 0

    # Calculate GLI where the denominator is not zero
    if np.any(valid):
        numerator = (2 * green) - red - blue
        gli[valid] = numerator[valid] / denominator[valid]

    logger.info(f"Adding GLI to channel {index + 1} (index: {index})")

    return gli


def calculate_exr(channels, index):
    red, green, blue = channels[..., 0], channels[..., 1], channels[..., 2]

    # Calculate the denominator
    denominator = red + green + blue

    # set the denominator to 1 if it is 0
    denominator[denominator == 0] = 1

    # calculate the numerator
    numerator = 1.4 * red - green

    # calculate the exr

    exr = numerator / denominator

    # scale from -1 to 1.4 to 0 to 1
    scaled_exr = (exr + 1) / 2.4

    logger.info(f"Adding Normalized EXR to channel {index + 1} (index: {index})")

    return scaled_exr


def passthrough_channel(
    channels, new_index, channel_index, transformation=None, print_debug=False
):
    logger.info(
        f"Mapping channel {channel_index + 1} (index: {channel_index}) to {new_index + 1} (index: {new_index})"
    )

    if print_debug:
        # loop over the x, y, z dimensions and print the value of the channel
        for x in range(channels.shape[0]):
            for y in range(channels.shape[1]):
                for z in range(channels.shape[2]):
                    value = channels[x, y, z, channel_index]

                    if value > 0:
                        logger.info(f"({x}, {y}, {z}): {value}")

    channel = channels[..., channel_index]

    if transformation is not None:
        channel = transformation(channel)

    return channel


def normalize_channel(channel):
    # Rescale the channel to the range [0, 1]
    min_val = np.min(channel)
    max_val = np.max(channel)
    return (channel - min_val) / (max_val - min_val)


def calculate_normalized_difference(channels, index):
    red, green = channels[..., 0], channels[..., 1]

    denominator = green + red

    ngd = np.zeros_like(denominator, dtype=float)

    valid = denominator != 0

    if np.any(valid):
        numerator = green - red
        ngd[valid] = numerator[valid] / denominator[valid]

    logger.info(
        f"Adding Normalized Difference Green/Red to channel {index + 1} (index: {index})"
    )

    return ngd


def filter_outliers(channels):
    # lets do this on the z axis to start
    # we are looking to loop over x[:,:, z, :] and remove any outliers by setting density (:3) to 0

    # for each z layer, calculate number of non-zero entries in the 4th dimension across 1:3
    # add each layer until it reaches 5th percentile of total non-zero entries
    # remove layers below this threshold

    # Calculate the total number of non-zero entries in the 4th dimension
    total_nonzero_entries = np.count_nonzero(channels[..., :3] > 0)
    z_dim = channels.shape[2]
    x_dim = channels.shape[0]
    y_dim = channels.shape[1]

    logger.info("Total non-zero entries:", total_nonzero_entries)

    fifth_percentile = 10

    cumulative_nonzero_entries = 0

    for z in range(z_dim):
        for x in range(x_dim):
            for y in range(y_dim):
                if np.any(channels[x, y, z, :3] > 0):
                    cumulative_nonzero_entries += 1
                    if cumulative_nonzero_entries >= fifth_percentile:

                        if z - 1 <= 0:
                            return channels

                        return channels[:, :, z - 1 :, :]

    return channels
