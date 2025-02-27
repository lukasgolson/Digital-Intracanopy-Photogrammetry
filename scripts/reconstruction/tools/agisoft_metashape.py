from __future__ import annotations  # Must be at the very top.

import logging
from enum import Enum
from pathlib import Path
from typing import List, Union, Tuple

from scripts.Utilities.dir_helpers import get_all_files
from scripts.Utilities.check_metashape_status import check_agisoft_license


from loguru import logger


try:
    import Metashape
    from Metashape import Chunk, Camera
except ImportError:
    logger.warning("Agisoft Metashape is not installed. Using mock objects.")
    from unittest.mock import MagicMock

    Metashape = MagicMock(name="Metashape")

import laspy
import numpy as np
from PIL import Image
from tqdm import tqdm

import scripts.reconstruction.metaflux as mf

from scripts.Utilities.sliding_window import sliding_window
from scripts.Utilities.string_helpers import get_numeric_from_string

from scripts.reconstruction.config import SensorConfig


class MatchingAccuracy(Enum):
    """Enum class for matching accuracy levels in Agisoft Metashape.

    The value defines the downscaling factor by each side applied to the original image.
    The only exception is Highest matching accuracy, where the images are upscaled by
    two times by each side.

    For matching accuracy the correspondence should be the following:
    Highest = 0
    High = 1
    Medium = 2
    Low = 4
    Lowest = 8

    For depth maps quality:
    Ultra = 1
    High = 2
    Medium = 4
    Low = 8
    Lowest = 16

    https://www.agisoft.com/forum/index.php?topic=11697.0


    """
    HIGHEST = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 4
    LOWEST = 8

class ItrAlignmentAlgorithm(Enum):
    NONE = 0
    V1 = 1
    V2 = 2

def get_alignment_version_enum(version: int | None) -> ItrAlignmentAlgorithm:
    if version is None:
        version = 0

    if version < 0 or version > 2:
        version = 0
    return ItrAlignmentAlgorithm(version)

def export_pointcloud(chunk: Metashape.Chunk, pointcloud_path: Path, tiepoints_path: Path):
    chunk.exportPointCloud(
        str(pointcloud_path.resolve()),
        source_data=Metashape.DataSource.PointCloudData,
    )
    chunk.exportPointCloud(
        str(tiepoints_path.resolve()),
        source_data=Metashape.DataSource.TiePointsData,
    )


class MetashapePipeline:

    def __init__(self, data_path: Path, frames_path: Path, mask_path: Union[Path, None] = None,
            export: [Path, None] = None, set_size: int = 30 * 5, quality_threshold: float = 0.6,
            user_input: bool = False, image_resolution: Tuple[int, int] = (1920, 1080),
            keypoint_count: int = 1600000, tiepoint_count: int = 0, chunk_count: int = 1,
            iterative_match_enabled: bool = False, subset_match_cloud_enabled: bool = False,
            task_subdivision: bool = False, iterative_alignment_version: int | None = None,
            config_dir: Path = Path("config"),
            camera_buffer_ratio: float = 0.25, camera_percentile=0):

        self.Doc: [Metashape.Document, None] = None
        self.Chunks: [List[Metashape.Chunk], None] = None
        self.ChunkImageScaleFactors: [List[int], None] = []
        self.Loaded: bool = False
        self.data_path: Path = data_path
        self.frames_path: Path = frames_path
        self.mask_path: Path = mask_path
        self.quality_threshold: float = quality_threshold
        self.iterative_match_enabled = iterative_match_enabled

        self.iterative_alignment_algorithm: ItrAlignmentAlgorithm = get_alignment_version_enum(
            iterative_alignment_version)
        self.chunkCount = chunk_count

        self.config_dir: Path = data_path / config_dir

        if export is None:
            logger.warning("No export path provided. Using data path.")
        else:
            self.export_path: Path = export

        self.set_size: int = set_size
        self.alignment_count: int = 0
        self.requires_user_input = user_input
        self.target_image_resolution: Tuple[int, int] = image_resolution
        self.keypointCount = keypoint_count
        self.tiepointCount = tiepoint_count
        self.camera_percentile = camera_percentile
        self.camera_buffer_ratio = camera_buffer_ratio

        self.subset_match_cloud_enabled = subset_match_cloud_enabled
        self.task_subdivision = task_subdivision

        self.SensorConfig = SensorConfig(config_dir)
        self.SensorConfig.read_config()

        self.pipeline = self.setup_pipeline()

    def setup_pipeline(self):
        mf.logger.setLevel(logging.INFO)


        tasks = [
            self.add_frames_task,
            self.load_masks_task,
            self.detect_markers_task,
            self.print_camera_count_task,
            self.clean_cameras_task,
            self.match_photos_task,
            self.broad_align_cameras_task,
            self.iterative_align_cameras_task,
            self.filter_cameras_task,
            self.subset_matches_task,
            self.build_rough_model_task,
            self.reduce_overlap_task,
            self.merge_chunks_task,
            self.build_depth_maps_task,
            self.build_point_cloud_task,
            self.export_camera_poses_task,
            self.export_point_cloud_task
        ]


        pipeline = mf.Pipeline(max_retries=2).add_task(tasks)
        pipeline.add_task(self.create_or_load_metashape_project_task, task_type=mf.TaskType.INIT)
        pipeline.add_task(self.save_project_task, task_type=mf.TaskType.INTER)
        pipeline.add_task(self.print_camera_count_task, task_type=mf.TaskType.INTER)

        logger.debug(pipeline.get_pipeline_string())

        return pipeline

    def run(self):
        self.pipeline.run(resume_filepath=self.export_path / "pipeline_state.pkl")

    def create_or_load_metashape_project_task(self) -> bool:
        self.Doc, self.Chunks, self.Loaded = create_or_load_metashape_project(self.export_path)

        return True

    def save_project_task(self):
        self.Doc.save()
        return True

    def print_camera_count_task(self):
        for chunk in self.Chunks:
            logger.info(f"Enabled Camera count: {camera_enabled_count(chunk)}")
        return True

    def add_frames_task(self):
        logger.info("Adding frames to project.")
        frames_raw = get_all_files(self.frames_path, "*")
        frames_raw.sort(key=get_numeric_from_string)

        if not frames_raw:
            logger.warning(f"No frames found in the specified path: {self.frames_path}")
            return
        else:
            logger.info(f"Found {len(frames_raw)} frames in the specified path: {self.frames_path}")

        for frame_window in sliding_window(frames_raw, len(frames_raw) // self.chunkCount, 250):
            logger.info(f"Adding {len(frame_window)} frames to chunk.")
            chunk = self.Doc.addChunk()
            self.Chunks.append(chunk)
            add_frames_to_chunk(chunk, frame_window)

            self.configure_sensor(chunk.sensors[0])

            image_sizes = [Image.open(frame).size for frame in frame_window]
            actual_image_resolution = np.mean(image_sizes, axis=0)
            average_size_ratio_x = actual_image_resolution[0] / self.target_image_resolution[0]
            average_size_ratio_y = actual_image_resolution[1] / self.target_image_resolution[1]
            scale_factor_raw = min(average_size_ratio_x, average_size_ratio_y)
            scale_factor = 2 ** round(
                np.log2(scale_factor_raw)) if scale_factor_raw > 1 else 1 if scale_factor_raw == 1 else 0
            self.ChunkImageScaleFactors.append(scale_factor)

        return True

    def load_masks_task(self):
        if self.mask_path is not None:
            logger.info("Loading sky masks.")
            for chunk in self.Chunks:
                load_masks(chunk, self.mask_path)
        else:
            logger.info("No mask path provided. Skipping mask loading.")

        return True

    def detect_markers_task(self):
        logger.info("Detecting fiducial markers.")
        for chunk in self.Chunks:
            chunk.detectMarkers(tolerance=50)

        return True

    def clean_cameras_task(self):
        logger.info("Cleaning cameras.")
        for chunk in self.Chunks:
            remove_low_quality_cameras(chunk, self.quality_threshold)

        return True

    def match_photos_task(self):
        logger.info("Matching photos.")
        for i, chunk in enumerate(self.Chunks):
            scale_factor = self.ChunkImageScaleFactors[i]
            if self.iterative_match_enabled:
                logger.info("Iteratively matching photos.")
                iterative_match_photos(chunk, self.keypointCount, self.tiepointCount, 0.3, scale_factor, self.set_size,
                                       subdivide=self.task_subdivision)
            else:
                logger.info("Broadly matching photos.")
                chunk.matchPhotos(
                    cameras=chunk.cameras, downscale=scale_factor, generic_preselection=True,
                    filter_stationary_points=False, reference_preselection=False, filter_mask=True,
                    mask_tiepoints=True, reset_matches=False, keep_keypoints=True,
                    keypoint_limit=self.keypointCount, subdivide_task=self.task_subdivision,
                    tiepoint_limit=self.tiepointCount
                )

        return True

    def broad_align_cameras_task(self):
        for chunk in self.Chunks:
            logger.info("Broadly aligning cameras.")
            chunk.alignCameras(
                cameras=chunk.cameras, adaptive_fitting=True, reset_alignment=True,
                subdivide_task=self.task_subdivision
            )

        return True

    def iterative_align_cameras_task(self):
        if self.iterative_alignment_algorithm is not ItrAlignmentAlgorithm.NONE:
            for chunk in self.Chunks:
                if self.iterative_alignment_algorithm is ItrAlignmentAlgorithm.V2:
                    logger.info("Iteratively aligning cameras using the V2 algorithm.")
                    iterative_align_cameras_v2(chunk, self.set_size, self.task_subdivision)
                else:
                    logger.info("Iteratively aligning cameras using the V1 algorithm.")
                    iterative_align_cameras(chunk, self.set_size, self.task_subdivision)

            self.alignment_count += 1
        else:
            logger.info("Skipping iterative alignment as it is not enabled.")

        return True

    def subset_matches_task(self):
        logger.info("Sub-setting sparse cloud")

        if self.subset_match_cloud_enabled:
            for chunk in self.Chunks:
                precount = get_tiepointcloud_count(chunk)

                subset_chunk_by_cameras(chunk, self.camera_percentile, self.camera_buffer_ratio)

                postcount = get_tiepointcloud_count(chunk)

                logger.info(f"Sparse cloud subsetting completed. Removed {precount - postcount} points.")

        return True

    def export_camera_poses_task(self):
        logger.info("Exporting camera positions")
        chunk = self.get_merged_chunk()
        if not chunk:
            logger.error("No chunk available to get camera positions from.")
            return

        return export_cameras(chunk, self.export_path / "camera_positions.las")

    def filter_cameras_task(self):
        for chunk in self.Chunks:
            logger.info("Current point count: " + str(get_tiepointcloud_count(chunk)))
            passes = 3
            initial_point_count = get_tiepointcloud_count(chunk)
            minimum_points = initial_point_count * 0.5
            for i in range(passes):
                if get_tiepointcloud_count(chunk) < minimum_points:
                    logger.info(f"Point count is less than {minimum_points}. Stopping filtering.")
                    break

                projection_accuracy = 48 * ((passes - i) + 1)
                reprojection_error = 1 * ((passes - i) + 1)

                reconstruction_uncertainty = 100 * ((passes - i) + 1)
                filters = [
                    (Metashape.TiePoints.Filter.ProjectionAccuracy, projection_accuracy),
                    (Metashape.TiePoints.Filter.ReprojectionError, reprojection_error),
                    (Metashape.TiePoints.Filter.ReconstructionUncertainty, reconstruction_uncertainty)
                ]

                logger.info(f"Filtering tiepoints. Pass {i + 1} of {passes}")
                try:
                    iterative_filter(chunk, filters=filters, reduction_ratio=0.75)
                except Exception as e:
                    logger.error(f"An error occurred while filtering tiepoints: {e}. Continuing...")
                logger.info("Current point count: " + str(get_tiepointcloud_count(chunk)))

        return True



    def build_rough_model_task(self):
        for chunk in self.Chunks:
            chunk.buildDepthMaps(downscale=4, filter_mode=Metashape.NoFiltering, max_neighbors=100, reuse_depth=False)
            chunk.buildModel(
                surface_type=Metashape.SurfaceType.Arbitrary,
                source_data=Metashape.DataSource.DepthMapsData,
                interpolation=Metashape.Interpolation.DisabledInterpolation,
                subdivide_task=self.task_subdivision
            )

        return True

    def reduce_overlap_task(self):
        for chunk in self.Chunks:
            chunk.reduceOverlap(overlap=10)

        return True

    def merge_chunks_task(self):
        if len(self.Chunks) > 1:
            logger.info("Merging chunks")
            self.Doc.alignChunks(chunks=self.Chunks, reference=self.Chunks[0], method=2, fit_scale=True)
            self.Doc.mergeChunks(chunks=self.Chunks, merge_tiepoints=True, merge_depth_maps=True)
            logger.info("Chunks merged")

        return True

    def configure_sensor(self, sensor):

        config_section = self.SensorConfig.get_section("sensor")

        shutter_model_map = {
            "disabled":    Metashape.Shutter.Model.Disabled,
            "regularized": Metashape.Shutter.Model.Regularized,
            "full":        Metashape.Shutter.Model.Full,
        }

        sensor_type_map = {  # Frame, Fisheye, Spherical, Cylindrical, RPC
            "frame":       Metashape.Sensor.Type.Frame,
            "fisheye":     Metashape.Sensor.Type.Fisheye,
            "spherical":   Metashape.Sensor.Type.Spherical,
            "cylindrical": Metashape.Sensor.Type.Cylindrical,
            "rpc":         Metashape.Sensor.Type.RPC,

        }

        # Get the configuration value and set the shutter model / sensor type
        shutter_model_config_value = config_section.get("shutter", "regularized")
        sensor_type_config_value = config_section.get("type", "frame")

        logger.info(f"Setting sensor shutter model to {shutter_model_config_value}...")
        logger.info(f"Setting sensor type to {sensor_type_config_value}...")

        sensor.rolling_shutter = shutter_model_map.get(shutter_model_config_value, Metashape.Shutter.Model.Regularized)
        sensor.type = sensor_type_map.get(sensor_type_config_value, Metashape.Sensor.Type.Frame)

        if config_section.get("load_calibration_file", False):
            config_path = self.config_dir / config_section.get("calibration_file", "calibration.xml")

            logger.info(f"Loading calibration file from {config_path}...")

            sensor.user_calib = Metashape.Calibration()
            sensor.user_calib.load(str(config_path.resolve()))

        if config_section.get("enable_sensor_correction", False):
            sensor.pixel_height = config_section.get("pixel_height_mm", 0.0019)
            sensor.pixel_width = config_section.get("pixel_width_mm", 0.0019)

            logger.info(f"Setting sensor pixel height-width to {sensor.pixel_height}-{sensor.pixel_width}...")

    def build_depth_maps_task(self):
        chunk = self.get_merged_chunk()
        chunk.buildDepthMaps(
            downscale=2, filter_mode=Metashape.NoFiltering, max_neighbors=64,
            cameras=chunk.cameras, subdivide_task=self.task_subdivision
        )

        return True

    def build_point_cloud_task(self):
        chunk = self.get_merged_chunk()
        chunk.buildPointCloud(
            source_data=Metashape.DataSource.DepthMapsData, point_confidence=True,
            keep_depth=True, uniform_sampling=False, points_spacing=0.1,
            subdivide_task=self.task_subdivision
        )

        return True

    def export_point_cloud_task(self):
        chunk = self.get_merged_chunk()
        export_pointcloud(chunk, self.export_path / "pointcloud.las", self.export_path / "tiepoints.las")

        return True

    def get_merged_chunk(self):
        chunks = self.Doc.chunks
        return get_merged_chunk(chunks)


def camera_position(chunk: Metashape.Chunk):
    t = chunk.transform.matrix
    crs = chunk.crs
    return [(crs.project(t.mulp(cam.center))) for cam in chunk.cameras if cam.transform]


def camera_enabled_count(chunk: Metashape.Chunk):
    return len([cam for cam in chunk.cameras if cam.enabled])


def camera_bounding_box(chunk, percentile: int = 0, buffer_ratio=0.15):
    camera_positions = camera_position(chunk)
    if not camera_positions:
        return None

    x_positions = [pos.x for pos in camera_positions]
    y_positions = [pos.y for pos in camera_positions]
    z_positions = [pos.z for pos in camera_positions]

    percentile_max = 100 - percentile
    min_x = np.percentile(x_positions, percentile)
    max_x = np.percentile(x_positions, percentile_max)
    min_y = np.percentile(y_positions, percentile)
    max_y = np.percentile(y_positions, percentile_max)
    min_z = np.percentile(z_positions, percentile)
    max_z = np.percentile(z_positions, percentile_max)

    diff_x = max_x - min_x
    diff_y = max_y - min_y
    diff_z = max_z - min_z

    buffer_x = diff_x * buffer_ratio
    buffer_y = diff_y * buffer_ratio
    buffer_z = diff_z * buffer_ratio

    min_x -= buffer_x
    max_x += buffer_x
    min_y -= buffer_y
    max_y += buffer_y
    min_z -= buffer_z
    max_z += buffer_z

    bounding_box = Metashape.BBox()
    bounding_box.min = Metashape.Vector([min_x, min_y, min_z])
    bounding_box.max = Metashape.Vector([max_x, max_y, max_z])

    return bounding_box


def set_reconstruction_region_from_bounding_box(chunk, bounding_box):
    region = chunk.region
    transform_matrix = chunk.transform.matrix
    region.size = bounding_box.max - bounding_box.min
    region.center = bounding_box.min + region.size / 2
    region.rot = transform_matrix.rotation().t()
    chunk.region = region


def trim_chunk_by_cameras(chunk, bounding_box):
    points = chunk.tie_points.points

    for point in points:
        if not (bounding_box.min.x <= point.coord.x <= bounding_box.max.x and
                bounding_box.min.y <= point.coord.y <= bounding_box.max.y and
                bounding_box.min.z <= point.coord.z <= bounding_box.max.z):
            point.selected = True
        else:
            point.selected = False

    chunk.tie_points.removeSelectedTracks()
    chunk.tie_points.removeSelectedPoints()


def subset_chunk_by_cameras(chunk, camera_percentile=1, camera_buffer_ratio=0.15):
    bounding_box = camera_bounding_box(chunk, percentile=camera_percentile, buffer_ratio=camera_buffer_ratio)

    if bounding_box is None:
        return

    logger.info(f"Bounding box: {bounding_box.min}, {bounding_box.max}")

    chunk.optimizeCameras()
    trim_chunk_by_cameras(chunk, bounding_box)
    set_reconstruction_region_from_bounding_box(chunk, bounding_box)


def export_cameras(chunk: Metashape.Chunk, export_file_path: Path):
    camera_positions = camera_position(chunk)
    if not camera_positions:
        logger.error("No camera positions available.")
        return False

    # if export_file_path does not have .las extension, add it to the end
    export_file_path = export_file_path.with_suffix(".las")

    las = laspy.create(point_format=3, file_version="1.2")
    las.x = [pos.x for pos in camera_positions]
    las.y = [pos.y for pos in camera_positions]
    las.z = [pos.z for pos in camera_positions]
    las.write(str(export_file_path.resolve()))

    logger.info(f"Camera positions successfully exported to {export_file_path}")

    return True


def get_merged_chunk(chunks):
    for chunk in chunks:
        if chunk.label == "Merged Chunk":
            return chunk
    return chunks[0]


def create_or_load_metashape_project(data: Path) -> Tuple[Metashape.Document, List[Metashape.Chunk], bool]:
    project_path = data / "metashape_project.psx"
    project_path = project_path.resolve()
    doc = Metashape.Document()
    loaded = False
    doc.read_only = False

    if project_path.exists():
        logger.info("Loading existing project")
        doc.open(str(project_path), read_only=False, ignore_lock=True)
        chunks = doc.chunks
        loaded = True
    else:
        logger.info("Creating new project")
        doc.save(path=str(project_path))
        doc.read_only = False
        chunks = doc.chunks

    return doc, chunks, loaded


def add_frames_to_chunk(chunk: Metashape.Chunk, frames: List[Path]):
    logger.info(f"Adding {len(frames)} frames to chunk")
    chunk.addPhotos([str(path) for path in frames])


def load_masks(chunk: Metashape.Chunk, mask_path: Path):
    mask_file_path = str(mask_path.resolve()) + "/{filename}_mask.png"

    try:
        chunk.generateMasks(
            path=mask_file_path, masking_mode=Metashape.MaskingModeFile
        )
    except Exception as e:
        logger.error(f"An error occurred while generating masks: {e}. Continuing...")


def handle_error(e: Exception):
    logger.error(f"An error occurred: {e}")


def iterative_match_photos(chunk, keypoint_count: int, tiepoint_count: int = 0, overlap_ratio: float = 0.3,
        scale_factor: float = 1, set_size=(30 * 20), subdivide=False):
    """
       Iteratively matches photos in the given chunk by attempting to match a set of cameras at a time. The process
       continues until all sets of cameras have been matched.

       :param keypoint_count: The number of keypoints to use for matching.
       :param chunk: The Metashape Chunk to perform operations on.
       :param overlap_ratio: The overlap between sets of cameras. Defaults to 0.3.
       :param scale_factor: The downscale factor to apply to the images. Defaults to 1. Minimum value is equal to 25 images.
       :param set_size: Number of cameras to match at a time. Defaults to 30 * 5. Minimum is 50.
       :return:
       """
    total_images = len(chunk.cameras)
    set_size = min(max(set_size, 50), total_images)
    overlap = max(int(set_size * overlap_ratio), 25)
    total_sets = (total_images // set_size) + (total_images % set_size > 0)

    logger.info(f"Total number of sets for matching: {total_sets}")
    logger.info(f"scale factor: {scale_factor}")

    with tqdm(total=total_sets, desc="Matching Set", unit="set", dynamic_ncols=True, position=0) as pbar:
        for i in range(0, total_images, set_size):
            match_list = list()
            start_index = max(0, i - overlap)
            end_index = min(total_images, i + set_size)

            for j, camera in enumerate(chunk.cameras[start_index:end_index]):
                match_list.append(camera)

            logger.info(f"Matching cameras {start_index} to {end_index} out of {total_images} cameras.")
            try:
                chunk.matchPhotos(
                    cameras=match_list, downscale=scale_factor, generic_preselection=True,
                    filter_stationary_points=False, reference_preselection=False, filter_mask=True,
                    mask_tiepoints=True, reset_matches=False, keep_keypoints=True,
                    keypoint_limit=keypoint_count, subdivide_task=subdivide, tiepoint_limit=tiepoint_count
                )
            except Exception as e:
                handle_error(e)

            pbar.update(1)


def align_cameras(chunk: Chunk, cameras: List[Camera], subdivide=False) -> None:
    try:
        chunk.alignCameras(cameras=cameras, subdivide_task=subdivide, adaptive_fitting=True, reset_alignment=True)
    except Exception as error:
        handle_error(error)


def iterative_align_cameras_v2(chunk: Chunk, batch_size: int = 50, max_iterations: int = 5, subdivide=False) -> None:
    """
      Iteratively aligns cameras in the given chunk by attempting to optimize camera alignment multiple times.
      :param chunk: The chunk containing cameras to be aligned.
      :param batch_size: The number of cameras to align in each batch.
      :param max_iterations: The maximum number of iterations to perform.
      :param subdivide: Whether to enable fine-level task subdivision.
      :return: None
    """
    for iteration in range(max_iterations):
        realign_batch = []

        for index, camera in tqdm(enumerate(chunk.cameras), total=len(chunk.cameras), desc="Processing cameras"):
            if camera.transform is None:
                realign_batch.append(camera)

                for post_index in range(index + 1, min(index + batch_size, len(chunk.cameras))):
                    post_camera = chunk.cameras[post_index]
                    if post_camera.transform is None:
                        realign_batch.append(post_camera)
                    else:
                        break

                if len(realign_batch) >= batch_size:
                    align_cameras(chunk, realign_batch, subdivide)
                    realign_batch = []

        if realign_batch:
            align_cameras(chunk, realign_batch, subdivide)


def iterative_align_cameras(chunk: Chunk, batch_size: int = 50, max_iterations: int = 5, timeout: int = 2,
        subdivide: bool = False) -> None:
    """
        Tries to realign cameras in the given chunk by attempting to optimize camera alignment multiple times.
        It stops if the process isn't improving as set by the timeout or if it reaches a maximum iteration count.

        :param chunk: The Metashape Chunk.
        :param batch_size: The number of cameras to be realigned per iteration. Defaults to 50.
        :param max_iterations: Maximum number of iterations to perform. Defaults to 5.
        :param timeout: Maximum number of stale iterations to perform before finishing. Defaults to 2.
        :param subdivide: Whether to enable fine-level task subdivision. Defaults to False.
    """
    iteration = 0
    stale_iterations = 0
    pbar_iteration = tqdm(total=max_iterations, desc="Realignment Iterations", dynamic_ncols=True)

    while iteration < max_iterations:
        num_unaligned_cameras_start = get_unaligned_cameras(chunk)
        realign_batch = []

        for camera in tqdm(chunk.cameras, desc="Processing cameras", dynamic_ncols=True):
            if camera.transform is None:
                realign_batch.append(camera)
                if len(realign_batch) >= batch_size:
                    align_cameras(chunk, realign_batch, subdivide)
                    realign_batch = []
            elif realign_batch:
                align_cameras(chunk, realign_batch, subdivide)
                realign_batch = []

        if realign_batch:
            align_cameras(chunk, realign_batch, subdivide)

        num_unaligned_cameras_end = get_unaligned_cameras(chunk)

        if num_unaligned_cameras_end >= num_unaligned_cameras_start:
            stale_iterations += 1
        else:
            stale_iterations = 0

        if stale_iterations >= timeout:
            logger.info(f"Stopping realignment after {timeout} stale iterations.")
            break

        iteration += 1
        pbar_iteration.update()

    pbar_iteration.close()
    logger.info(f"Realignment completed after {iteration} iterations.")


def get_unaligned_cameras(chunk):
    return sum(camera.transform is None for camera in chunk.cameras)


def approximate_new_threshold(mean, std, threshold=0.5, strictness=0.5):
    """
    Calculate a new threshold for image quality removal by combining base values
    and statistical adjustments.

    Parameters:
    - mean: The average quality score of the images.
    - std: The standard deviation of the quality scores.
    - threshold: The base threshold value (default is 0.5).
    - strictness: Determines how aggressively the threshold is adjusted based on the scores (default is 0.5).

    Returns:
    - The new threshold value, ensuring it is not below 0.
    """
    adjusted_threshold = mean + strictness * (mean - std)
    base_threshold = (threshold + mean) / 2
    weight = mean / (mean + std)
    comp_weight = 1 - weight
    approx_threshold = comp_weight * adjusted_threshold + weight * base_threshold

    return max(0, approx_threshold)


def remove_low_quality_cameras(chunk: Chunk, threshold: float = 0.5) -> None:
    chunk.analyzeImages(filter_mask=True)
    cameras_to_remove = []

    quality_scores = [float(camera.meta['Image/Quality']) for camera in chunk.cameras]
    mean = np.mean(quality_scores)
    sigma = np.std(quality_scores)  # Calculate standard deviation for the approximate function
    new_threshold = approximate_new_threshold(mean, sigma, threshold)

    logger.info(
        f"Threshold of {threshold}, but mean of {mean}. Removing cameras with quality scores below {new_threshold}.")
    for camera in tqdm(chunk.cameras, desc=f"Scanning for low quality cameras (threshold of {new_threshold})",
                       dynamic_ncols=True):
        quality = float(camera.meta['Image/Quality'])
        if quality < new_threshold:
            cameras_to_remove.append(camera)

    logger.info(f"Removing {len(cameras_to_remove)} low quality cameras")
    chunk.remove(cameras_to_remove)


def get_tiepointcloud_count(chunk: Metashape.Chunk):
    if chunk.tie_points.points is None:
        return 0
    return len(chunk.tie_points.points)


def iterative_filter(chunk: Metashape.Chunk, filters: list, reduction_ratio=0.75, iterations: int = -1) -> None:
    """
       Iteratively optimizes the alignment of tie points in a Metashape Chunk. This is accomplished by removing points
       with high error based on multiple criteria and adjusting camera positions. The iterative process continues until
       the error falls below specified thresholds or the maximum number of iterations is reached.

       :param filters: A list of tuples, each containing a filter criterion and its target max error.
       :param reduction_ratio: The ratio by which the error is reduced in each iteration.
       :param chunk: The Metashape Chunk to be optimized.
       :param iterations: Maximum number of iterations for the optimization process. If set to -1, the number of
       iterations is calculated based on the current reconstruction error. Defaults to -1. If set to a positive integer,
       the number of iterations is fixed.
    """
    logger.info("Optimizing alignment with filters and reduction ratio of " + str(reduction_ratio))
    points = chunk.tie_points.points

    if points is None:
        logger.info("No tie points found")
        return

    tiepoint_filters = []
    for criterion, target_max_error in filters:
        tiepoint_filter = Metashape.TiePoints.Filter()
        tiepoint_filter.init(chunk, criterion=criterion)
        tiepoint_filters.append((tiepoint_filter, target_max_error))

    if iterations < 0:
        iteration_counts = []
        for tiepoint_filter, target_max_error in tiepoint_filters:
            reconstruction_error = [error for i, error in enumerate(tiepoint_filter.values) if points[i].valid]
            starting_error = np.max(reconstruction_error)
            iterations_for_criterion = int(np.log(target_max_error / starting_error) / np.log(reduction_ratio))
            iteration_counts.append(iterations_for_criterion)

            logger.info(
                f"Criterion requires {iterations_for_criterion} iterations."
                f"Current error: {starting_error}, target error: {target_max_error}")

            iterations = max(iteration_counts)
            logger.info(f"Using the largest number of iterations required: {iterations}")

            if iterations <= 0:
                logger.info("All criteria have reached their target error. Skipping optimization.")
                return

    for _ in tqdm(range(iterations), desc="Optimizing alignment"):
        max_errors = []
        mean_errors = []
        target_max_error = 0
        for tiepoint_filter, target_max_error in tiepoint_filters:
            reconstruction_error = [error for i, error in enumerate(tiepoint_filter.values) if points[i].valid]

            max_error = np.max(reconstruction_error)
            mean_error = np.mean(reconstruction_error)
            max_errors.append(max_error)
            mean_errors.append(mean_error)
            target_error = max_error * reduction_ratio

            if max_error <= target_max_error:
                logger.info(f"Criterion {tiepoint_filter.criterion} has reached target error of {target_max_error}")
                continue

            if target_error < target_max_error:
                target_error = target_max_error

            tiepoint_filter.selectPoints(target_error)
            chunk.tie_points.removeSelectedPoints()

        chunk.optimizeCameras(adaptive_fitting=True)
        if all(error <= target_max_error for error in max_errors):
            break


def process_frames(data: Path, frames_path: Path, export_path: Path, mask_path: Union[Path, None] = None,
        set_size: int = 30 * 5, quality_threshold: float = 0.75, user_input: bool = False,
        image_resolution: Tuple[int, int] = (3840, 2160), keypoint_count: int = 1600000, tiepoint_count=0,
        chunk_count=1, iterative_match_enabled: bool = False, iterative_alignment_version: int | None = 0,
        task_subdivision=False):

    export_path.mkdir(parents=True, exist_ok=True)

    pipeline = MetashapePipeline(
        data, frames_path, mask_path, export_path, set_size=set_size,
        quality_threshold=quality_threshold, user_input=user_input, image_resolution=image_resolution,
        keypoint_count=keypoint_count, tiepoint_count=tiepoint_count, chunk_count=chunk_count,
        iterative_match_enabled=iterative_match_enabled, iterative_alignment_version=iterative_alignment_version,
        task_subdivision=task_subdivision
    )
    pipeline.run()


def activate_license(license_key: str):
    """

    Args:
        license_key:

    Returns:    :bool: Whether the license should be deactivated after the operation.

    """
    is_installed, is_licensed = check_agisoft_license()

    if not is_installed:
        logger.error("Agisoft Metashape is not installed.")
        return False, False

    if not is_licensed:
        logger.info("Activating Agisoft Metashape license.")
        import Metashape
        Metashape.license.activate(license_key)
        license_activated = True
    else:
        logger.info("Agisoft Metashape is already licensed.")
        license_activated = False

    return license_activated


def deactivate_license():
    import Metashape
    Metashape.license.deactivate()
