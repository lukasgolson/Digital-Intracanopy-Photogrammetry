import os
from pathlib import Path

from autokit import ExternalTool, ToolConfig, PlatformData
from autokit.ExecutableType import ExecutableType
from loguru import logger


class Me2Net(ExternalTool):
    def __init__(self, base_dir: Path = "./third-party"):
        super().__init__(base_dir=base_dir, lazy_setup=True, progress_bar=True)

    @property
    def config(self) -> ToolConfig:

        data = PlatformData(
            url="https://github.com/lukasgolson/me2net/archive/refs/heads/main.zip",
            subdir=Path("me2net-main"),
            executable=Path("me2net.py"))

        return ToolConfig(
            tool_name="Me2Net",
            platform_data={
                'Windows': data,
                'Darwin': data,
                'Linux': data},
            executable_type=ExecutableType.PYTHON
        )

    def remove_background(self, source: Path, destination: Path):
        """
       Remove sky from images using SkyRemoval tool.

       :param self: Instance reference
       :param source: The source image path. This can be a single image or a directory.
       :param destination: The destination directory path where the processed images will be stored.
       :raises Exception: If there is a failure in executing the command to the Me2Model tool.
       """

        if not destination.exists():
            os.mkdir(destination)

        command = _form_background_removal_command(source, destination)
        try:
            exit_code = self.run_command(command)
            logger.success(f"Me2 completed with exit code: {exit_code}")
        except Exception as e:
            logger.error(f"Failed to execute the command: {e}")
            raise




def _form_background_removal_command(source: Path, destination: Path, mode: int = 2) -> str:
    """Form the command string to pass to the Background removal tool
    :rtype: str
    """
    if mode < 0 or mode > 2:
        raise ValueError(f"Invalid mode: {mode}; mode must be 0, 1, or 2.")

    command = f'dir -mu {mode} "{source.resolve()}" "{destination.resolve()}"'

    return command
