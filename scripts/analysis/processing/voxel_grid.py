import logging

import numpy as np
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


class VoxelGrid:
    def __init__(self, shape=None, existing_array=None):
        """
        Initializes a new instance of a ColorVoxelGrid. Can either create a new grid
        with the specified shape or wrap around an existing 4D numpy array.
        :param shape: A tuple of three integers, the dimensions of the voxel grid if a new grid is created.
                      This is ignored if existing_array is provided.
        :param existing_array: An optional 4D numpy array to use as the grid.
        """
        if existing_array is not None:
            assert (
                    existing_array.ndim == 4
            ), "Existing array must be 4D with the last dimension being for channels."
            self.grid = existing_array
        else:
            assert (
                    shape is not None
            ), "Shape must be provided if no existing array is given."
            self.grid = np.zeros(shape + (3,), dtype=np.uint8)

        self.scaling_factor = 1

    def get_color(self, x, y, z):
        """
        Retrieves the color at a specific coordinate in the grid.
        :param x: The x-coordinate.
        :param y: The y-coordinate.
        :param z: The z-coordinate.
        :return: The RGB color tuple at the specified coordinates.
        """
        return self.grid[x, y, z]

    def set_color(self, x, y, z, color):
        """
        Sets the color at a specific coordinate in the grid.
        :param x: The x-coordinate.
        :param y: The y-coordinate.
        :param z: The z-coordinate.
        :param color: The RGB color tuple to set.
        """
        self.grid[x, y, z] = color

    def calculate_slice(self, z):
        """
        Displays a 2D slice of the grid at a given z-coordinate.
        :param z: The z-coordinate of the slice to display.
        """
        table = self.grid[:, :, z, 0:3]
        table[table == 0] = np.nan
        return table

    def calculate_convex_hull_volume(self, threshold=0):
        """
        Calculates the volume of the convex hull of the voxel grid where brightness is above a threshold.
        :param threshold: Brightness threshold to filter the voxels.
        :return: Volume of the convex hull.
        """
        brightness = np.linalg.norm(self.grid, axis=3)
        x, y, z = np.where(brightness > threshold)
        points = np.column_stack((x, y, z))

        if points.shape[0] < 4:
            logger.debug("Not enough points to create a convex hull. Returning 0.")
            return 0

        hull = ConvexHull(points)
        return hull.volume

    def save_grid(self, filename, x_offset=0, y_offset=0, z_offset=0):
        """
        Saves the voxel grid to a CSV file, including color information for each voxel.

        :param filename: The name of the file to save the voxel grid.
        :param x_offset: Offset to apply to the x-coordinate of each voxel.
        :param y_offset: Offset to apply to the y-coordinate of each voxel.
        :param z_offset: Offset to apply to the z-coordinate of each voxel.
        """
        # Get the dimensions of the grid
        x_dim, y_dim, z_dim, _ = self.grid.shape

        # Prepare data for CSV
        rows = []

        # Iterate through the grid to collect voxel positions and colors
        for x in range(x_dim):
            for y in range(y_dim):
                for z in range(z_dim):
                    # Check if the voxel has color information (non-zero)
                    if np.any(self.grid[x, y, z, 0:3]):  # Check for non-black voxel
                        # Adjust the voxel position by the offsets
                        adjusted_x = x + x_offset
                        adjusted_y = y + y_offset
                        adjusted_z = z + z_offset

                        # Get the color information
                        color = self.grid[x, y, z, 0:3]

                        # multiply the color by 255 to get the RGB values
                        color = color * 255

                        # Append adjusted position and color to the rows
                        rows.append([adjusted_x, adjusted_y, adjusted_z, color[0], color[1], color[2]])

        # Write to CSV file
        with open(filename, "w") as file:
            # Write header
            file.write("x,y,z,r,g,b\n")
            # Write voxel data
            for row in rows:
                file.write(",".join(map(str, row)) + "\n")

        logger.info(f"Voxel grid saved to {filename}")
