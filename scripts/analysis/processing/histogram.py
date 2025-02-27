import numpy as np
from scipy.signal import savgol_filter

def calculate_canopy_live_crown(
    filtered_grid,
    channel=3,
    threshold=0,
    base_height_percentile: int = 5,
    top_height_percentile: int = 95,
    smoothing_window: int = 5,
    derivative_threshold_fraction: float = 0.1,
):

    z_layers = filtered_grid.shape[2]
    pixel_counts = np.zeros(z_layers)

    # Collect pixel counts for each z layer
    for z in range(z_layers):
        slice_z = filtered_grid[:, :, z, channel]
        valid_pixels = (slice_z > threshold) & (~np.isnan(slice_z))
        pixel_counts[z] = np.sum(valid_pixels)

    # Smooth the pixel counts using Savitzky-Golay filter
    # Ensure the window size is odd and less than the number of layers
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    if smoothing_window >= z_layers:
        smoothing_window = z_layers - 1 if z_layers % 2 == 1 else z_layers - 2

    smoothed_counts = savgol_filter(pixel_counts, smoothing_window, polyorder=2)

    # Compute the first derivative
    derivative = np.diff(smoothed_counts)

    # Find the base of the canopy (significant positive increase)
    max_positive_derivative = np.max(derivative)
    positive_derivative_threshold = (
        max_positive_derivative * derivative_threshold_fraction
    )

    base_of_canopy_index = None
    for z in range(1, len(derivative)):
        if derivative[z] > positive_derivative_threshold:
            base_of_canopy_index = z
            break

    if base_of_canopy_index is None:
        raise ValueError("No valid base of canopy found.")

    # Find the top of the canopy (significant negative decrease)
    max_negative_derivative = np.min(derivative)
    negative_derivative_threshold = (
        max_negative_derivative * derivative_threshold_fraction
    )

    top_of_canopy_index = None
    for z in range(len(derivative) - 1, 0, -1):
        if derivative[z] < negative_derivative_threshold:
            top_of_canopy_index = z + 1  # Adjust index because of np.diff
            break

    if top_of_canopy_index is None:
        raise ValueError("No valid top of canopy found.")

    # Use height_percentiles to refine the base and top of canopy
    # Base of canopy
    valid_pixels_heights_base = []
    for z in range(base_of_canopy_index, z_layers):
        slice_z = filtered_grid[:, :, z, channel]
        valid_pixels = (slice_z > threshold) & (~np.isnan(slice_z))
        if np.any(valid_pixels):
            valid_pixels_heights_base.extend([z] * np.sum(valid_pixels))

    if valid_pixels_heights_base:
        valid_pixels_heights_base = np.array(valid_pixels_heights_base)
        base_of_canopy = np.percentile(
            valid_pixels_heights_base, base_height_percentile
        )
    else:
        raise ValueError(
            "No valid pixels found for base height percentile calculation."
        )

    # Top of canopy
    valid_pixels_heights_top = []
    for z in range(0, top_of_canopy_index + 1):
        slice_z = filtered_grid[:, :, z, channel]
        valid_pixels = (slice_z > threshold) & (~np.isnan(slice_z))
        if np.any(valid_pixels):
            valid_pixels_heights_top.extend([z] * np.sum(valid_pixels))

    if valid_pixels_heights_top:
        valid_pixels_heights_top = np.array(valid_pixels_heights_top)
        top_of_canopy = np.percentile(valid_pixels_heights_top, top_height_percentile)
    else:
        raise ValueError("No valid pixels found for top height percentile calculation.")

    return base_of_canopy, top_of_canopy
