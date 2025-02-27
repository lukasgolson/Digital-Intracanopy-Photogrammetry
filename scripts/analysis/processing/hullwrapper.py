import logging
from pathlib import Path

import numpy as np
from numpy._typing import NDArray
from scipy.spatial import ConvexHull, QhullError

from scripts.analysis.processing.voxel_grid import VoxelGrid

logger = logging.getLogger(__name__)


class HullWrapper(object):

    defined: bool = False
    equations: NDArray[np.float64]
    vertices: NDArray[np.float64]  # Store vertices of the hull
    faces: NDArray[np.int32]  # Store faces of the hull

    def __init__(self):
        self.defined = False
        self.equations = None
        self.vertices = None
        self.faces = None

    def get_equations(self):
        if self.defined:
            return self.equations
        else:
            return None

    def save_hull(self, filename: str | Path, x_offset=0, y_offset=0, z_offset=0):
        """
        Saves the hull (vertices and faces) to a file in a simple format.

        :param filename: The name of the file to save the hull.
        :param x_offset: Offset to apply to the x-coordinate of each vertex.
        :param y_offset: Offset to apply to the y-coordinate of each vertex.
        :param z_offset: Offset to apply to the z-coordinate of each vertex.
        """
        if self.defined:
            try:
                with open(filename, 'w') as f:
                    # Write vertices with offsets
                    for vertex in self.vertices:
                        adjusted_vertex = [
                            vertex[0] + x_offset,
                            vertex[1] + y_offset,
                            vertex[2] + z_offset,
                        ]
                        f.write(f"v {' '.join(map(str, adjusted_vertex))}\n")

                    # Write faces (1-based indexing)
                    for face in self.faces:
                        f.write(f"f {' '.join(map(lambda x: str(x + 1), face))}\n")

                logger.info(f"Hull saved successfully to {filename}.")
            except Exception as e:
                logger.error(f"Error saving hull: {e}")
        else:
            logger.error("Hull is not defined, cannot save.")

    def calculate_hull_from_grid(self, grid: VoxelGrid, threshold=0):
        """
        Calculates the convex hull of the voxel grid where brightness is above a threshold.

        :param grid: The ColorVoxelGrid object containing voxel data.
        :param threshold: Brightness threshold to filter the voxels.
        """
        x_size, y_size, z_size, channels = grid.grid.shape

        non_empty_indices = []
        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    if np.any(grid.grid[x, y, z, :3] > threshold):  # Consider only RGB channels
                        non_empty_indices.append((x, y, z))

        points = np.array(non_empty_indices)

        if points.shape[0] < 4:
            logger.warning(f"Only {points.shape[0]} points in voxel grid. Not enough points to create a convex hull. Creating an empty mesh.")
            self.defined = True
            self.vertices = np.array([])
            self.faces = np.array([])
            self.equations = np.array([])
            return

        try:
            hull = ConvexHull(points)
            self.vertices = hull.points  # Get vertices from the hull
            self.faces = hull.simplices  # Get faces from the hull
        except QhullError as e:
            logger.error(f"Error creating convex hull: {e}")
            hull = None

        if hull is not None:
            self.defined = True
            self.equations = hull.equations  # Optional: save equations if needed
        else:
            self.defined = False