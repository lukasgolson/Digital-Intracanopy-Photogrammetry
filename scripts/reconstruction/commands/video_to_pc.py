import datetime
import os
import shutil
import time
import traceback
from pathlib import Path

from dyncfg import DynamicConfig, ConfigValue
from loguru import logger

from scripts.Utilities.dir_helpers import get_leaf_directories, get_all_files, create_flat_folder_name
from scripts.reconstruction.tools import irss_media_tools, ModelExecutionEngines
from scripts.reconstruction.tools.me2net import Me2Net
from scripts.reconstruction.tools.agisoft_metashape import process_frames, activate_license
from scripts.Utilities.string_helpers import format_elapsed_time


def process_videos(config_path: Path = "reconstruction.ini"):
    """
    Function to process files from a specified directory for tree reconstruction
    """

    config = DynamicConfig(config_path)

    data_dir = config['Directories']['data_dir'].or_default("data").as_path()
    video_parent_dir = config['Directories']['video_path'].or_default(
        (data_dir / "video").relative_to(data_dir.parent)).as_path()
    root_export_path = config['Directories']['export_path'].or_default(
        (data_dir / "export").relative_to(data_dir.parent)).as_path()

    temp_path = config['Directories']['temp_path'].or_default(
        (data_dir / "temp").relative_to(data_dir.parent)).as_path()

    frames_path_root = config['Directories']['frames_path'].or_default(
        (temp_path / "frames").relative_to(data_dir.parent)).as_path()

    mask_path_root = config['Directories']['mask_path'].or_default(
        (temp_path / "masks").relative_to(data_dir.parent)).as_path()

    log_path_root = config['Directories']['log_path'].or_default(
        (data_dir / "logs").relative_to(data_dir.parent)).as_path()

    tools_path = config['Directories']['tools_path'].or_default(
        (data_dir / "tools").relative_to(data_dir.parent)).as_path()

    regenerate = config['Settings']['regenerate_tmp_output'].or_default("False").as_bool()
    user_input = config['Settings']['user_input'].or_default("False").as_bool()
    process_individually = config['Settings']['process_individually'].or_default("False").as_bool()
    debug = config['Settings']['debug'].or_default("False").as_bool()

    drop_ratio = config['Frames']['drop_ratio'].or_default(0.95).as_float()
    quality_threshold = config['Frames']['quality_threshold'].or_default(0.5).as_float()

    use_mask = config['Frames']['use_mask'].or_default("False").as_bool()
    mask_model = config['Frames']['mask_model'].or_default("Skymask")
    mask_processor = config['Frames']['mask_processor'].or_default("gpu")

    image_resolution = config['Frames']['image_resolution'].or_default("3840x2160").apply(
        replace_in_strings, ["(", ")"]).as_list("x").as_int()[:2]

    assert len(image_resolution) == 2, "Image resolution must be a tuple of two integers"

    image_resolution = (image_resolution[0], image_resolution[1])

    iterative_match_enabled = config['Processing']['iterative_match_enabled'].or_default("False").as_bool()
    iterative_alignment_enabled = config['Processing']['iterative_alignment_enabled'].or_default("False").as_bool()

    if iterative_alignment_enabled:
        iterative_alignment_version = config['Processing']['iterative_alignment_version'].or_default(1).as_int()
    else:
        iterative_alignment_version = None
    alignment_set_length = config['Processing']['alignment_set_length'].or_default(3 * 50).as_int()

    keypoint_count = config['Processing']['keypoint_count'].or_default(1600000).as_int()
    tiepoint_count = config['Processing']['tiepoints_count'].or_default(0).as_int()
    subdivision_enabled = config['Processing']['tasksubdivision'].or_default("false").as_bool(False)

    chunk_count = config['Chunking']['chunk_count'].or_default(1).as_int()

    should_manage_license = config['Settings']['manage_metashape_license'].or_default("False").as_bool()
    metashape_license = config['Settings']['metashape_license'].or_default("")

    deactivate_license_when_done = False

    if should_manage_license:
        deactivate_license_when_done = activate_license(metashape_license)

    from scripts.Utilities.check_metashape_status import check_agisoft_license
    is_installed, is_licensed = check_agisoft_license()

    if is_licensed:
        logger.info("Agisoft Metashape Professional License is active. ")
    else:
        logger.error("Agisoft Metashape Professional license is not active.")
        return

    if not data_dir.exists():
        logger.warning(f"The provided data path: {data_dir} does not exist.")
        return

    if not video_parent_dir.exists():
        logger.warning(f"The provided video path: {video_parent_dir} does not exist.")
        return


    if regenerate:
        clean_directories(root_export_path, temp_path)


    # Discover all the leaf directories in the data-video directory
    video_leaf_dirs = get_leaf_directories(video_parent_dir)

    logger.info("Number of footage_group_subdirs: " + str(len(video_leaf_dirs)))
    logger.debug("Processing video files in the following subdirectories:")
    for leaf_dir in video_leaf_dirs:
        logger.debug(f"{leaf_dir.resolve()}")

    # Group the video files in the leaf directories into either individual clips or the entire directory
    # based on the process_individually flag
    input_paths = []
    for leaf_dir in video_leaf_dirs:
        if process_individually:
            individual_clips = get_all_files(leaf_dir, "*.mp4")
            input_paths += individual_clips
        else:
            input_paths.append(leaf_dir)

    beginning_time = time.time()

    project_logger = None
    for input_path in input_paths:

        file_start_time = time.time()

        if project_logger is not None:
            logger.remove(project_logger)

        if input_path.is_file():
            logger.debug(f"Processing video file: {input_path.resolve()}")
        else:
            logger.debug(f"Processing video files in {input_path.resolve()}")

        flat_folder_name = create_flat_folder_name(input_path, video_parent_dir)

        project_logger = logger.add(log_path_root / flat_folder_name / "{time}.log",
                                    rotation="50 MB", compression="zip", backtrace=True, diagnose=debug, catch=True,
                                    watch=True)


        try:

            frames_path = frames_path_root / flat_folder_name
            mask_path = mask_path_root / flat_folder_name

            export_path = root_export_path / flat_folder_name

            input_video_files = collect_input_videos(input_path)

            if len(input_video_files) == 0:
                logger.error(f"No video files found in the provided path: {input_path}")
                return

            for file in input_video_files:
                logger.info(f"Processing file: {file.resolve()}")

            media_tools = irss_media_tools.MediaTools(base_dir=tools_path)

            if not frames_path.exists():
                extract_frames(media_tools, input_video_files, frames_path, drop_ratio)

            if use_mask:
                if not mask_path.exists():
                    mask_frames(media_tools, mask_model, mask_path, frames_path, tools_path, mask_processor)
            else:
                mask_path = None

            process_frames(data_dir, frames_path, export_path, mask_path, set_size=alignment_set_length,
                           quality_threshold=quality_threshold, user_input=user_input,
                           image_resolution=image_resolution, keypoint_count=keypoint_count,
                           tiepoint_count=tiepoint_count, chunk_count=chunk_count,
                           iterative_match_enabled=iterative_match_enabled,
                           iterative_alignment_version=iterative_alignment_version,
                           task_subdivision=subdivision_enabled)


        except Exception as e:

            error = (f"Failed to process video files in {input_path.resolve()}.\n"
                     f"Error: {e}")

            if debug:
                error += f"\nTraceback: {traceback.format_exc()}"

            logger.error(error)
            logger.info("Continuing with the next input...")

        end_time = time.time()
        frame_elapsed_time = end_time - file_start_time
        logger.debug(
            f"Input Execution time: {format_elapsed_time(frame_elapsed_time)}; Total elapsed time: {format_elapsed_time(end_time - beginning_time)}")

    if deactivate_license_when_done:
        from scripts.reconstruction.tools.agisoft_metashape import deactivate_license
        deactivate_license()


def mask_frames(media_tools, mask_model, mask_path, processing_path, tools_path, mask_processor):
    mask_path.mkdir(parents=True, exist_ok=True)

    if mask_model == "Me2Net":
        me2 = Me2Net(base_dir=tools_path)
        me2.remove_background(processing_path, mask_path)

        # for each file in the mask path, rename from image.png to image_mask.png
        for path in mask_path.rglob('*'):
            if path.is_file():
                new_name = path.stem + "_mask" + path.suffix
                path.rename(path.with_name(new_name))

    else:

        processor_mapping = {
            "gpu":  ModelExecutionEngines.TENSORRT,
            "cuda": ModelExecutionEngines.CUDA,
            "cpu":  ModelExecutionEngines.CPU
        }

        if mask_processor not in processor_mapping:
            mask_processor = "cpu"

        processors_order = ["cuda", "gpu", "cpu"]
        start_index = processors_order.index(mask_processor)
        processors_order = processors_order[start_index:]

        for processor in processors_order:
            try:
                media_tools.mask_sky(processing_path, mask_path, processor_mapping[processor])
                break
            except Exception:
                continue


def extract_frames(media_tools, input_files, output_dir, drop_ratio):
    frame_start_time = time.time()
    # Now recreate the directory
    output_dir.mkdir(parents=True, exist_ok=True)
    for video_file in input_files:
        media_tools.extract_frames(video_file, output_dir, drop_ratio)
    frame_end_time = time.time()
    frame_elapsed_time = frame_end_time - frame_start_time
    file_count = 0
    for path in output_dir.rglob('*'):
        if path.is_file():
            file_count += 1
    logger.debug(f"Extracted {file_count} frames in {format_elapsed_time(frame_elapsed_time)}")


def collect_input_videos(video_path):
    video_files = []
    if os.path.isfile(video_path):
        video_files = [video_path]
    elif os.path.isdir(video_path):
        suffices = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.wmv", "*.flv", "*.webm"]
        video_files = get_all_files(video_path, *suffices)
    return video_files


def clean_directories(export_path, *other_paths):

    for path in other_paths:
        if path.is_dir():
            shutil.rmtree(path)
            logger.info(f"Deleted directory {path.resolve()}")

    if export_path.is_dir():
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        new_path = export_path.parent / f"{export_path.name}_old{timestamp}"
        os.rename(export_path, new_path)
        logger.info(f"Renamed {export_path} to {new_path.resolve()}")


def replace_in_strings(text: ConfigValue | str, old_strings, new="") -> ConfigValue:
    if isinstance(old_strings, str):
        old_strings = [old_strings]

    for old in old_strings:
        text = text.replace(old, new)

    return text