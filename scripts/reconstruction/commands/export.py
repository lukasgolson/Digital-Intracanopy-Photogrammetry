from __future__ import annotations

from pathlib import Path
from loguru import logger

from scripts.reconstruction.tools.agisoft_metashape import get_merged_chunk, export_pointcloud, export_cameras

try:
    import Metashape
    from Metashape import Chunk, Camera
except ImportError:
    from unittest.mock import MagicMock
    Metashape = MagicMock(name="Metashape")

def build_dense_cloud(doc: Metashape.Document) -> bool:
    try:

        chunk = get_merged_chunk(doc.chunks)

        chunk.buildDepthMaps(quality=0, filter_mode=Metashape.NoFiltering, max_neighbors=200)
        chunk.buildPointCloud(
            source_data=Metashape.DataSource.DepthMapsData, point_confidence=True,
            keep_depth=True, uniform_sampling=False)

        doc.save()



    except Exception as e:
        logger.error(f"Failed to process dense point cloud. Details: {e}")
        return False


def export(project_path, export_path, cameras_only: bool, rebuild: bool = False):
    project_path = Path(project_path)
    export_path = Path(export_path)

    project_name = project_path.parent.stem

    try:
        logger.info(f"Processing {project_path}")
        doc = Metashape.Document()
        doc.open(path=str(project_path), read_only=False, ignore_lock=True)

        if rebuild:
            build_dense_cloud(doc)

        chunk = get_merged_chunk(doc.chunks)


        export_stem = Path(export_path) / project_name

        pcl_name = export_stem.with_name(export_stem.name + ".las")
        tiepoints_name = export_stem.with_name(export_stem.name + "_tiepoints.las")
        cameras_name = export_stem.with_name(export_stem.name + "_cameras.las")

        if not cameras_only:
            export_pointcloud(chunk, pcl_name, tiepoints_name)

        export_cameras(chunk, cameras_name)
    except Exception as e:
        logger.error(f"Failed to process {project_path}. Details: {e}")
        return False
    return True
