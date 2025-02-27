import hashlib
import logging
import os
from pathlib import Path

import numpy as np
from numba import njit, prange
from pyntcloud import PyntCloud

from scripts.analysis.processing.color_utilities import rgb_to_hsv
from scripts.analysis.processing.pointgrid import adjust_classification
from scripts.analysis.tools import Delineation

logger = logging.getLogger(__name__)


def get_file_hash(file_path):
    file_path = str(file_path)

    # load the file and calculate the hash
    with open(file_path, "rb") as file:
        file_data = file.read()
        file_hash = hashlib.md5(file_data).hexdigest()

    """Generate a hash for the given file path."""
    return file_hash


def clean_point_cloud(input_path, output_path):
    input_path = Path(input_path)

    output_path = Path(output_path)
    hash_file_path = f"{output_path}.hash"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_hash = get_file_hash(input_path)

    if os.path.exists(output_path):
        """Load and clean the point cloud, utilizing hash checking to avoid reprocessing."""
        if os.path.exists(hash_file_path):
            with open(hash_file_path, "r") as file:
                existing_hash = file.read()
            if existing_hash == file_hash:
                logger.debug(f"File with hash '{existing_hash}' already cleaned.")
                return True
    else:
        logger.debug(f"File '{output_path}' does not exist. Processing...")

        delineation_tool = Delineation()
        try:
            delineation_tool.run(input_path, output_path)
        except Exception as e:
            logger.error(f"Error processing file '{input_path}': {e}")
            return False

        # Save the hash file to indicate completion for future checks
        with open(hash_file_path, "w") as file:
            file.write(file_hash)

    return True


def load_point_cloud(rgb_pointcloud_path: Path, vertical_axis=2):
    """
    Load a point cloud from disk using either PDAL or PyntCloud.
    The point cloud is returned as a 2D NumPy array with the following columns:
    0-2: X, Y, Z coordinates;
    3-5: Red, Green, Blue color values;
    6: Classification (0 = unclassified, 1 = ground);
    7: Tree Group ID;
    :param rgb_pointcloud_path: Path to the point cloud file
    :param vertical_axis: The axis that represents the upward direction;
    :return: NumPy array with dimensions (N, 8) where N is the number of points in the point cloud
    """

    rgb_pointcloud_path = Path(rgb_pointcloud_path)

    logger.debug("Loading points from PyntCloud")


    pynt_cloud: PyntCloud = PyntCloud.from_file(
        rgb_pointcloud_path.resolve().as_posix()
    )
    coords = pynt_cloud.points[["x", "y", "z"]].values.astype(np.float32)
    colors = pynt_cloud.points[["red", "green", "blue"]].values.astype(np.float32)

    classification = pynt_cloud.points[["raw_classification"]].values.astype(
        np.float32
    )  # load classification

    tree_id = pynt_cloud.points[["treeid"]].values.astype(
        np.float32
    )  # load tree id

    # currently classification is either 0 or 2; but we could also have 0 2 3;
    # so we need to adjust the classification to start at 0; and each increasing value is a new classification;
    # for example, if we have 0, 2, 3, we would adjust it to 0, 1, 2
    # if we have 0 3 5, we would adjust it to 0 1 2

    # adjust classification to start at 0; find the minimum value and subtract it from the entire array
    classification = adjust_classification(classification)

    axis_order = {
        0: [1, 2, 0],  # X as vertical
        1: [0, 2, 1],  # Y as vertical
        2: [0, 1, 2],  # Z as vertical (or any other value)
    }

    if vertical_axis not in axis_order:
        vertical_axis = 2
        logger.warning(f"Invalid vertical axis value. Defaulting to Z-axis (2).")

    index_list = axis_order[vertical_axis]

    adjusted_pointcloud = coords[:, index_list]

    coords = np.column_stack((adjusted_pointcloud, colors, classification, tree_id))

    return coords


def mask_blue(pointcloud: np.ndarray):
    mask = create_color_mask(pointcloud, (180, 250), 0.2, 0.8, 0.3)
    return pointcloud[mask]


@njit(parallel=True)
def create_color_mask(pcl, hue_range=(180, 250), sat_low=0.2, sat_high=0.8, value_threshold=0.3):
    mask = np.ones(len(pcl), dtype=np.bool_)
    for i in prange(len(pcl)):
        r, g, b = pcl[i][3], pcl[i][4], pcl[i][5]
        h, s, v = rgb_to_hsv(r, g, b)
        if (
                min(hue_range) < h < max(hue_range)  # Hue in blue range
                or s < sat_low  # Too desaturated (near white)
                or s > sat_high  # Too saturated (pure color)
                or v < value_threshold  # Too dark
        ):
            mask[i] = False  # Remove this point
    return mask