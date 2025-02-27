from pathlib import Path

import click

from scripts.Utilities.dir_helpers import execute_on_each_file, get_all_subdir
from scripts.analysis.analysis import run_analysis
from scripts.analysis.processing.clip import clip_pointcloud

import logging
logger = logging.getLogger(__name__)


@click.command("predict", help="Extract features from tree point clouds.")
@click.argument("config_path", type=click.Path(exists=True), default="predict-config.ini")
def run_analysis_command(config_path: str):
    run_analysis(config_path)


@click.command("clip", help="Clips exported point clouds to their camera extents")
@click.argument("cloud_path", type=click.Path(exists=True), default="data/export")
@click.argument("export_path", type=click.Path(exists=False), default="data/clipped")
@click.argument("quantile", default=0)
@click.argument("buffer_percent", default=0.1)
@click.option("--ignore_z", is_flag=True, help="Ignore Z-values for convex hull used in clipping.")
def clip_command(cloud_path: str, export_path: str, quantile: int, buffer_percent: float, ignore_z: bool = True):

    if not Path(export_path).exists():
        Path(export_path).mkdir(parents=True)


    leaf_dirs = get_all_subdir(cloud_path)

    for leaf_dir in leaf_dirs:
        camera_las = leaf_dir / Path("camera_positions.las")
        point_cloud_las = leaf_dir / Path("pointcloud.las")
        output_las = Path(export_path) / Path(f"{leaf_dir.name}.las")

        # validate the existence of the camera and point cloud files
        if not camera_las.exists():
            print(f"Camera file not found: {camera_las}")
            continue

        if not point_cloud_las.exists():
            print(f"Point cloud file not found: {point_cloud_las}")
            continue

        clip_pointcloud(str(point_cloud_las.resolve()), str(camera_las.resolve()), str(output_las),
                        quantile, buffer_percent, ignore_z)

        logger.info(f"Clipped point cloud saved to {output_las}")



@click.group(name="analysis", help="All analysis commands.")
def analysis_cli():
    pass


analysis_cli.add_command(run_analysis_command)
analysis_cli.add_command(clip_command)