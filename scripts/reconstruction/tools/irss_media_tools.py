"""
Python Bindings for IRSSMediaTools.
"""
from abc import ABC
from pathlib import Path

import GPUtil
from autokit import ExternalTool, ToolConfig, PlatformData

from loguru import logger

from ..tools import ModelExecutionEngines


class MediaTools(ExternalTool, ABC):

    def __init__(self, base_dir: Path = "./third-party"):
        super().__init__(base_dir=base_dir, lazy_setup=True, progress_bar=True)

    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            tool_name="IRSSMediaTools",
            platform_data={
                'Windows': PlatformData(
                    url="https://github.com/IRSS-UBC/MediaTools/releases/download/latest/win-x64.zip",
                    subdir=Path(),
                    executable=Path("IRSSMediaTools.exe")
                ),
                'Darwin': PlatformData(
                    url="https://github.com/IRSS-UBC/MediaTools/releases/download/latest/osx-x64.zip",
                    subdir=Path(),
                    executable=Path("IRSSMediaTools")
                ),
                'Linux': PlatformData(
                    url="https://github.com/IRSS-UBC/MediaTools/releases/download/latest/linux-x64.zip",
                    subdir=Path(),
                    executable=Path("IRSSMediaTools")
                ), },

        )

    def run_command_logged(self, command: str) -> int:
        logger.info(f"Running command: {command}")

        # pipe the output of the command to the logger

        return self.run_command(command)

    def extract_frames(self, video_path: Path, output_path: Path, drop: float = 0.5) -> None:
        """
        Extract frames from the video_old at the given path.

        Args:
            video_path: The path to the video_old file.
            output_path: The path to save the extracted frames.
            drop: The drop rate for frame extraction.

        Returns:
            None
        """
        logger.info(f"Extracting frames from video_old at path: {video_path}")

        command = f'extract --input "{video_path.resolve()}" ' \
                  f'--output "{output_path.resolve()}" ' \
                  f'--drop {drop} --format png'

        try:
            exit_code = self.run_command_logged(command)

            if exit_code != 0:
                raise Exception(f"Failed to extract frames from video_old at path: {video_path}")
            else:
                logger.success(
                    f"Extracting frames from video_old at path: {video_path} completed with exit code: {exit_code}")

        except Exception as e:
            logger.error(
                f"Exception occurred while extracting frames from video_old at path: {video_path}. Details: {e}")
            raise e

    def mask_sky(self, images: Path, output_path: Path,
                 engine: ModelExecutionEngines = ModelExecutionEngines.CPU, gpu_count: int = 0) -> None:

        logger.info(f"Masking sky in images from path: {images}")

        detected_gpu_count = len(GPUtil.getAvailable(order='first', limit=8, maxLoad=0.75, maxMemory=0.75))

        if engine == ModelExecutionEngines.CUDA or engine == ModelExecutionEngines.TENSORRT:
            if gpu_count == 0:
                gpu_count = detected_gpu_count

            if gpu_count > detected_gpu_count:
                logger.warning(
                    f"Requested GPU count {gpu_count} is greater than detected GPU count {detected_gpu_count}.")

            if gpu_count <= 1:
                engine = ModelExecutionEngines.CPU
                logger.warning("No unused GPU(s) detected, falling back to CPU for masking")
                gpu_count = 1

        command = f'mask_sky --input "{images.resolve()}" ' \
                  f'--output "{output_path.resolve()}" ' \
                  f'--engine "{engine.value}" ' \
                  f'--gpu-count {gpu_count}'

        try:
            exit_code = self.run_command_logged(command)

            if exit_code != 0:
                raise Exception(f"Failed to mask images from path: {images}")
            else:
                logger.success(f"Masking images from path: {images} completed with exit code: {exit_code}")

        except Exception as e:
            logger.error(f"Exception occurred while masking images from path: {images}. Details: {e}")
            raise e
