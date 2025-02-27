import csv
import logging

from dyncfg import DynamicConfig

from scripts.analysis.collect_results import combine_results
from scripts.analysis.processing.filtering import apply_filters_and_masks
from scripts.analysis.processing.histogram import (
    calculate_canopy_live_crown,
)
from scripts.analysis.processing.hullwrapper import HullWrapper
from scripts.analysis.processing.scope_timer import ScopeTimer
from scripts.analysis.processing.voxel_grid import VoxelGrid

logger = logging.getLogger(__name__)

from typing import Optional, Any

from alive_progress import alive_it



from numba import njit
from numba.typed import List


from scripts import analysis
from scripts.analysis.processing.channel_limit import ChannelLimit
from scripts.analysis.processing.pointcloud import (
    clean_point_cloud,
    load_point_cloud,
    mask_blue
)

# Local imports
from scripts.analysis.processing.pointgrid import accumulate_points
from scripts.analysis.processing.ray_marching import process_rays, generate_rays, save_rays

from scripts.analysis.tools.threedf import ThreeDF

import numpy as np

current_debug_dir = ""

debug = False


@njit
def get_nonzero_indices(filtered_grid):
    """Extracts the indices of all non-zero entries in the grid."""
    return np.nonzero(filtered_grid)


def calculate_tree_height(filtered_grid, percentile=None):
    """Calculate tree height based on highest non-empty voxel on the z axis"""
    """If percentile is provided, calculate the height based on the percentile value
     of the non-empty voxels on the z axis"""
    indices = get_nonzero_indices(filtered_grid)

    # 99th percentile z
    if percentile is not None:
        max_z = np.percentile(indices[2], percentile)
        return max_z
    else:
        max_z = np.max(indices[2])
    return max_z


def calculate_canopy_dimensions(filtered_grid):
    """Calculates canopy width and length from non-zero voxels' min and max x and z coordinates."""
    indices = get_nonzero_indices(filtered_grid)
    min_x, max_x = np.min(indices[0]), np.max(indices[0])
    min_y, max_y = np.min(indices[1]), np.max(indices[1])
    width = max_x - min_x
    length = max_y - min_y

    # return width as the larger of the two dimensions; length as the smaller
    if width < length:
        return length, width

    return width, length


def filter_channel_by_limits(
        grid: np.ndarray, channel_limits: Optional[List[ChannelLimit]] = None
) -> np.ndarray:
    if channel_limits is None or len(channel_limits) == 0:
        return grid

    if grid.size == 0:
        logger.warning("Data array is empty. Returning original data.")
        return grid

    masks = []

    for limit in channel_limits:
        channel = limit.channel
        min_val = limit.min_val
        max_val = limit.max_val
        mask_na = limit.mask_na

        if mask_na:
            na_mask = ~np.isnan(grid[:, :, :, channel])
            masks.append(na_mask)

        if min_val is None and max_val is None:
            if not mask_na:
                logger.warning(
                    f"Channel {channel} has no min or max value provided. Creating mask for NA values only."
                )
            continue

        if min_val is None:
            if limit.inclusive:
                channel_mask = grid[:, :, :, channel] >= max_val
            else:
                channel_mask = grid[:, :, :, channel] < max_val
        elif max_val is None:
            if limit.inclusive:
                channel_mask = grid[:, :, :, channel] <= min_val
            else:
                channel_mask = grid[:, :, :, channel] > min_val
        else:
            if limit.inclusive:
                channel_mask = (grid[:, :, :, channel] <= min_val) | (
                        grid[:, :, :, channel] >= max_val
                )
            else:
                channel_mask = (grid[:, :, :, channel] > min_val) & (
                        grid[:, :, :, channel] < max_val
                )

        masks.append(channel_mask)

    if not masks:
        logger.warning("No valid masks were created. Returning original data.")
        return grid

    combined_mask = np.all(masks, axis=0)
    grid[~combined_mask] = 0

    return grid


def canopy_transparency(
        voxel_grid,
        canopy_base_height,
        num_rays: int = 3000,
        channel_limits=None,
        hull_brightness_min: int = 0,
        random_z: bool = True,
        should_debug: bool = True,
        hit_requirement: int = 2,
        energy_threshold: float = 0.1,
        ray_dissipation_factor: float = 0.5,
        initial_energy: float = 1.0,
        step_size: float = 0.5,
):
    if channel_limits is None:
        channel_limits = []

    stem_mask = np.arange(voxel_grid.shape[2]) >= canopy_base_height

    # capture the hull before we do our filtering
    vg_with_stem = VoxelGrid(existing_array=voxel_grid)

    voxel_grid = voxel_grid[:, :, stem_mask, :]
    vg_canopy = VoxelGrid(existing_array=voxel_grid)

    hull = HullWrapper()
    hull.calculate_hull_from_grid(vg_canopy, hull_brightness_min)

    voxel_grid = filter_channel_by_limits(voxel_grid, channel_limits)

    vg = VoxelGrid(existing_array=voxel_grid)

    if debug and should_debug:
        # save the vg with stem to disk
        vg_with_stem.save_grid(Path(current_debug_dir) / "grid_with_stem.csv")

        # save the hull to disk as a obj file
        logger.info("Saving hull to disk")
        hull.save_hull(
            Path(current_debug_dir) / "hull.obj", z_offset=canopy_base_height
        )

        vg.save_grid(
            Path(current_debug_dir) / "grid_filtered.csv", z_offset=canopy_base_height
        )
        vg_canopy.save_grid(
            Path(current_debug_dir) / "grid_full.csv", z_offset=canopy_base_height
        )

    shape = np.shape(vg.grid)

    rays = generate_rays(
        shape[:3],
        num_rays,
        0,
        random_z,
    )

    # save rays to disk as a text file in the output
    if debug and should_debug:
        name = f"rays.txt"
        save_rays(
            rays,
            str(Path(current_debug_dir) / name),
            z_offset=canopy_base_height,
        )

    scope = ScopeTimer("Ray Marching", False)

    with scope:
        unobstructed_count, valid_ray_count = process_rays(
            voxel_grid,
            rays,
            hull,
            min_hit_requirement=hit_requirement,
            initial_ray_energy=initial_energy,
            energy_threshold=energy_threshold,
            ray_dissipation_factor=ray_dissipation_factor,
            step_size=step_size,
        )

    logger.info(
        f"Ray Marching took {round(scope.time_in_ms / num_rays, 2)} ms per ray. {round((valid_ray_count / num_rays) * 100, 2)}% of rays passed hull. {round((unobstructed_count / num_rays) * 100, 2)}% of rays were unobstructed."
    )

    if valid_ray_count == 0:
        logger.error("No valid rays output. Verify input data.")
        frac = -1
    else:
        frac = unobstructed_count / valid_ray_count

    result = frac * 100

    return result


def process_pointcloud(input_file, outputDir, vertical_axis=2, voxel_size=0.1,
        config: DynamicConfig = DynamicConfig("config.ini")):
    data = {}
    data["Input"] = input_file.parent.name + "/" + input_file.name
    data["Voxel Size"] = voxel_size

    should_calculate_dbh = (
        config["analyses"]["calculate_dbh"].or_default(True).log().as_bool()
    )

    if should_calculate_dbh:
        dbh, height_3df = Run3DF(input_file, outputDir)
        data["DBH"] = dbh
        data["Height 3df"] = height_3df

    output_file = outputDir / "cleaned.las"

    should_mask_blue = config["filtering"]["mask_blue"].or_default(True).log().as_bool()
    filter_accumulate_points = (
        config["filtering"]["rgb_outlier_filter"].or_default(True).log().as_bool()
    )
    outlier_percentage = (
        config["filtering"]["rgb_outlier_percentage"].or_default(5).log().as_int()
    )

    cleaned = clean_point_cloud(
        input_file, output_file
    )
    if not cleaned:
        raise Exception(
            "Failed to clean point cloud. This is likely due to an issue with the input file."
        )

    # Load the cleaned point cloud
    pointcloud = load_point_cloud(output_file, vertical_axis=vertical_axis)
    logger.info("Loaded point cloud")

    if should_mask_blue:
        pointcloud = mask_blue(pointcloud)

    # Process point cloud into grid
    point_grid = accumulate_points(
        pointcloud, voxel_size, filter_accumulate_points, outlier_percentage
    )

    # unload the pointcloud to free up memory
    del pointcloud

    density_threshold = config["filtering"]["density_threshold"].or_default(0).log().as_float()

    # Filtering and analysis
    filtered_grid = apply_filters_and_masks(point_grid, density_threshold)

    logger.info("Analyzing canopy...")

    should_calculate_height = (
        config["analyses"]["calculate_height"].or_default(True).log().as_bool()
    )

    if should_calculate_height:
        try:
            tree_height = calculate_tree_height(filtered_grid)
        except Exception as e:
            logger.error(f"Error calculating tree height: {e}")
            tree_height = -1, -1

        data["Height"] = tree_height * voxel_size

    should_calculate_canopy_dims = (
        config["analyses"]["calculate_dimensions"].or_default(True).log().as_bool()
    )

    if should_calculate_canopy_dims:
        try:
            canopy_width, canopy_length = calculate_canopy_dimensions(filtered_grid)
        except Exception as e:
            logger.error(f"Error calculating canopy dimensions: {e}")
            canopy_width, canopy_length = -1, -1

        data["CWidth"] = canopy_width * voxel_size
        data["CLength"] = canopy_length * voxel_size

    smoothing_window = (
        config["live_crown_id"]["smoothing_window"]
        .or_default(5)
        .log()
        .as_int()
    )

    should_calculate_transparency = (
        config["analyses"]["calculate_transparency"].or_default(True).log().as_bool()
    )
    should_calculate_live_canopy = (
        config["analyses"]["calculate_live_canopy_dims"].or_default(True).log().as_bool()
    )


    live_canopy_base = -1
    live_canopy_top = -1

    # We can calculate the live canopy base and top if we are calculating transparency or live canopy
    if should_calculate_live_canopy or should_calculate_transparency:

        live_canopy_base, live_canopy_top = calculate_canopy_live_crown(filtered_grid, channel=3)

        data["Live Canopy Base"] = live_canopy_base * voxel_size
        data["Live Canopy Top"] = live_canopy_top * voxel_size

    if should_calculate_transparency:
        logger.info("Ray-Marching Canopy")

        sample_count = (
            config["transparency"]["ray_sample_count"].or_default(15000).log().as_int()
        )

        canopy_transparency_randomize_ray_trajectory = (
            config["transparency"]["randomize_ray_trajectory"]
            .or_default(True)
            .log()
            .as_bool()
        )

        brightness_floor = (
            config["transparency"]["brightness_floor"].or_default(0).log().as_int()
        )

        filter_non_photosynthetic = (
            config["transparency"]["filter_non_photosynthetic_voxels"]
            .or_default(True)
            .log()
            .as_bool()
        )

        channel_limit_grr = (
            config["transparency"]["filter_channel_limit_grr"]
            .or_default(0.005)
            .log()
            .as_float()
        )
        channel_limit_gli = (
            config["transparency"]["filter_channel_limit_gli"]
            .or_default(0)
            .log()
            .as_float()
        )

        channel_limits = []

        if filter_non_photosynthetic:
            channel_limits.append(
                ChannelLimit(5, channel_limit_grr, None, False)
            )

            channel_limits.append(
                ChannelLimit(3, channel_limit_gli, None, False)
            )  # GLI

        required_hits = config["transparency"]["required_hits"].or_default(2).log().as_int()
        energy_threshold = (
            config["transparency"]["energy_threshold"].or_default(0.1).log().as_float()
        )
        ray_dissipation_factor = (
            config["transparency"]["ray_dissipation_factor"]
            .or_default(0.5)
            .log()
            .as_float()
        )
        initial_energy = (
            config["transparency"]["initial_energy"].or_default(1).log().as_float()
        )

        transparency = canopy_transparency(
            filtered_grid,
            live_canopy_base,
            sample_count,
            channel_limits=channel_limits,
            hull_brightness_min=brightness_floor,
            random_z=canopy_transparency_randomize_ray_trajectory,
            should_debug=debug,
            hit_requirement=required_hits,
            energy_threshold=energy_threshold,
            ray_dissipation_factor=ray_dissipation_factor,
            initial_energy=initial_energy,
        )

        data["Transparency Sample"] = sample_count

        data[f"Transparency"] = transparency

    # for each numeric value, round to 3 decimal places
    for key, value in data.items():
        if isinstance(value, (int, float)):
            data[key] = round(value, 3)

    return data


def Run3DF(input_file, output_path):
    tdf = ThreeDF()
    try:
        dbh, height_3df = tdf.run(input_file, output_path / "3df", Path("config.ini"))
    except Exception as e:
        logger.info(e)
        dbh = -1
        height_3df = -1
    return dbh, height_3df


import os
from pathlib import Path


def calculate_directory_pairs(
        input_base_dir, output_base_dir, extension: str = None, paths_to_skip=None
):
    input_file_paths = []
    paths_to_skip = paths_to_skip or []  # Ensure it's always iterable

    # Walk through all directories and subdirectories in the base input directory
    for root, dirs, files in os.walk(input_base_dir, followlinks=True):
        for name in files:
            # Calculate relative path to maintain the directory structure in the output
            relative_path = os.path.relpath(root, input_base_dir)
            name_no_extension = Path(name).stem

            output_dir = Path(os.path.join(output_base_dir, relative_path))
            output_dir = (
                    output_dir.parent / output_dir.stem / name_no_extension
            ).resolve()

            input_file_path = Path(root) / name

            # Skip files if any of the paths_to_skip are present in the path
            if any(skip in str(input_file_path) for skip in paths_to_skip):
                continue

            # Skip files if extension is specified and doesn't match
            if extension is not None and input_file_path.suffix != extension:
                continue

            input_file_paths.append((input_file_path.resolve(), output_dir))

    return input_file_paths


def save_dict_as_csv(data: dict[str, Any], file_path: str | Path):
    fieldnames = data.keys()
    with open(file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(data)


def process_files(
        input_dir: Path,
        output_dir: Path,
        file_count: int = -1,
        paths_to_skip=None,
        extension=".las",
        vertical_axis=2,
        voxel_size=0.1,
        config: DynamicConfig = DynamicConfig("config.ini"),
):

    input_file_paths = calculate_directory_pairs(
        input_dir, output_dir, extension, paths_to_skip
    )

    if file_count > 0:
        input_file_paths = input_file_paths[:file_count]

    bar = alive_it(input_file_paths, title="Processing", unit=" file(s)")

    for file in bar:

        input_file_path = file[0]
        output_dir = file[1]

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Update the progress bar text
        # file parent directory
        parent_name = input_file_path.parent.name
        file_name = input_file_path.name
        display = f"{parent_name}/{file_name}"

        bar.text = f"'{display}'"

        try:
            global current_debug_dir
            current_debug_dir = Path(output_dir) / "debug"

            if not os.path.exists(current_debug_dir):
                os.makedirs(current_debug_dir)

            result = process_pointcloud(
                input_file_path, output_dir, vertical_axis, voxel_size, config
            )

            # print each data field
            for key, value in result.items():
                logger.info(f"{key}: {value}")

            # save results to a csv file; we are doing this per file to avoid losing data if the script crashes during processing or if the user stops the script early.
            # This has the added benefit of allowing the user to see the results as they come in rather than waiting for the entire script to finish
            # Alternatives include saving the results row by row, but this could add complexity to the script
            # We previously were appending to an array of dictionaries, but KISS and DRY principles apply here.

            csv_file_path = Path(output_dir.parent) / Path(output_dir.stem + ".csv")
            save_dict_as_csv(result, csv_file_path)

        except Exception as e:
            logger.warning(f"Error processing file: {input_file_path}")
            logger.exception(e)

        finally:
            logger.info(f"Finished processing file: {input_file_path}")


def run_analysis(config_file: str = "config.ini"):
    dc = DynamicConfig(config_file)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    global debug
    debug = dc["debug"]["debug"].or_default(True).log().as_bool()

    if debug:
        logger.log(logging.DEBUG, "Debug mode enabled.")
        logging.getLogger(DynamicConfig.__name__).setLevel(logging.INFO)

    input_base_dir = dc["data"]["input_dir"].or_default("data/input").log()
    output_base_dir = dc["data"]["output_dir"].or_default("data/output").log()
    final_output = dc["data"]["final_output"].or_default("data/results.csv").log()

    number_of_files_to_process = dc["data"]["file_count"].or_default(-1).log().as_int()

    vertical_axis = dc["data"]["vertical_axis"].or_default(2).log().as_int()

    voxel_size_config = (
        dc["voxelation"]["voxel_size"].or_default("0.1").as_list(",").as_float()
    )

    paths_to_skip = dc["data"]["paths_to_skip"].or_default("").log()

    if paths_to_skip == "":
        paths_to_skip = None
    else:
        paths_to_skip = paths_to_skip.split(",")

    for voxel_size in voxel_size_config:
        voxel_size = float(voxel_size)

        logger.info(f"Processing with voxel size: {voxel_size}")

        if len(voxel_size_config) > 1:
            output_base_dir_final = Path(output_base_dir) / f"voxel_{voxel_size}"
        else:
            output_base_dir_final = Path(output_base_dir)

        process_files(
            Path(input_base_dir),
            Path(output_base_dir_final),
            file_count=number_of_files_to_process,
            paths_to_skip=paths_to_skip,
            vertical_axis=vertical_axis,
            voxel_size=voxel_size,
            config=dc,
        )

    try:
        combine_results(output_base_dir, final_output)

    except Exception as e:
        logger.warning(
            "Don't panic.... No data has been lost, this is only an error combining results. Run collectResults.py to recover."
        )
        logger.exception(e)

    logger.info("Processing complete.")


if __name__ == "__main__":
    run_analysis()
