from pathlib import Path

import laspy
import numpy as np
from loguru import logger
from numba import njit, prange
from scipy.spatial import ConvexHull


@njit(parallel=True)
def is_within_hull_parallel(points, hull_equations, buf_distance):
    results = np.empty(len(points), dtype=np.bool_)
    for i in prange(len(points)):
        point = points[i]
        results[i] = np.all(np.dot(hull_equations[:, :-1], point.T) + hull_equations[:, -1] <= buf_distance)
    return results


@njit
def calculate_max_distance(vertices, points):
    max_distance = 0
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            dist = np.linalg.norm(points[vertices[i]] - points[vertices[j]])
            if dist > max_distance:
                max_distance = dist
    return max_distance


def clip_pointcloud(point_cloud_path: str, camera_cloud_path: str, output_file: str, quantile: int = 0,
                    buffer_percent: float = 0.1, ignore_z: bool = False):
    point_cloud_path = str(Path(point_cloud_path).resolve())
    camera_cloud_path = str(Path(camera_cloud_path).resolve())

    cloud_las = laspy.read(point_cloud_path)
    cloud_points = cloud_las.points.copy()
    cloud_xyz = np.vstack((cloud_las.x, cloud_las.y, cloud_las.z)).T

    camera_las = laspy.read(camera_cloud_path)
    camera_xyz = np.vstack((camera_las.x, camera_las.y, camera_las.z)).T

    min_quantile = quantile
    max_quantile = 100 - quantile

    camera_percentiles = np.percentile(camera_xyz, [min_quantile, max_quantile], axis=0)

    mask = np.all((camera_xyz >= camera_percentiles[0]) & (camera_xyz <= camera_percentiles[1]), axis=1)
    camera_cloud = camera_xyz[mask]

    if ignore_z:
        logger.info("Ignoring Z-values for convex hull.")
        camera_cloud_2d = camera_cloud[:, :2]  # Keep only X and Y
        hull = ConvexHull(camera_cloud_2d)
    else:
        hull = ConvexHull(camera_cloud)

    max_distance = calculate_max_distance(hull.vertices, camera_cloud[:, :2] if ignore_z else camera_cloud)

    buffer_distance = buffer_percent * max_distance

    hull_equations = hull.equations

    if ignore_z:
        mask = is_within_hull_parallel(cloud_xyz[:, :2], hull_equations, buffer_distance)
    else:
        mask = is_within_hull_parallel(cloud_xyz, hull_equations, buffer_distance)

    trimmed_points = cloud_points[mask]

    trimmed_las = laspy.create(point_format=cloud_las.header.point_format, file_version=cloud_las.header.version)
    trimmed_las.header = cloud_las.header
    trimmed_las.points = trimmed_points

    trimmed_las.write(output_file)
