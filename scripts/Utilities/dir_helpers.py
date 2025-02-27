import fnmatch
import re
from pathlib import Path
from typing import Union, List, Callable

import logging
logger = logging.getLogger(__name__)



def get_all_files(dir_path: Union[Path, str], *patterns: str) -> List[Path]:
    """
    This function retrieves all files in a given directory that match any of the provided patterns.

    :param dir_path: The directory path as a string or Path object where to look for files.
    :param patterns: One or more string patterns to match against the file names.
                     The patterns can include wildcards, such as '*' or '?'.
                     If multiple patterns are provided, they should be passed as separate arguments (not as a list).

    :return: A list of Path objects for the files that match any of the provided patterns.

    :raises TypeError: If any of the provided patterns is not a string.
    """

    dir_path = Path(dir_path)

    matching_files = set()

    for pattern in patterns:
        if not isinstance(pattern, str):
            raise TypeError(
                f"Expected string, got {type(pattern).__name__} instead. "
                f"If you are passing multiple patterns, make sure to use the * operator.")

        all_files = list(dir_path.rglob('*'))


        matching_files.update(
            [f for f in all_files if any(fnmatch.fnmatchcase(str(f).lower(), p.lower()) for p in patterns)])

    return list(matching_files)





def get_all_subdir(dir_path: Union[Path, str]) -> List[Path]:
    """
    Retrieve all subdirectories in the specified directory.
    :param dir_path: Directory path as a string or Path object
    :return: List of Path objects of all subdirectories
    """

    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    subdirs = [p for p in dir_path.iterdir() if p.is_dir()]

    return subdirs


def get_leaf_directories(dir_path: Union[Path, str], skip=True, skip_tag: str = "[skip]") -> List[Path]:
    """
    Retrieve all leaf directories (directories without subdirectories) in the specified directory.
    :param dir_path: Directory path as a string or Path object
    :param skip_tag: The tag used to identify directories to skip (default is "[skip]")
    :return: List of Path objects of all leaf directories
    """
    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    leaf_directories = []

    for p in dir_path.rglob('*'):

        if p.is_dir() and not any(c.is_dir() for c in p.iterdir()):
            # if subdir or its parent has name containing skip_tag, then skip it
            if skip and (skip_tag.lower() in str(p.resolve()).lower() or skip_tag.lower() in str(
                    p.parent.resolve()).lower()):
                continue

            leaf_directories.append(p)

    if dir_path.is_dir() and not any(c.is_dir() for c in dir_path.iterdir()):
        leaf_directories.append(dir_path)

    return leaf_directories


def create_flat_folder_name(dir_path: Union[Path, str], base_dir: Union[Path, str]) -> str:
    """
    Create a valid folder name for a flat version of the leaf directory's path.
    :param dir_path: The leaf directory path as a string or Path object
    :param base_dir: The base directory path as a string or Path object
    :return: A string representing the flattened folder name
    """

    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir

    relative_path = dir_path.relative_to(base_dir)

    flat_name = re.sub(r'[\\/:*?"<>|]', '_', str(relative_path))

    return flat_name



def execute_on_each_file(project_path: Path, suffices: List[str], function: Callable, *args, **kwargs) -> None:
    if not suffices:
        suffices = ["*"]

    if project_path.is_file():
        project_files = [project_path]
    elif project_path.is_dir():
        project_files = get_all_files(project_path, *suffices)
    else:
        logger.error(f"Invalid path: {project_path}")
        return

    if not project_files:
        logger.warning(f"No files found in: {project_path}")
        return

    for project_file in project_files:
        logger.info(f"Found: {project_file}")

    for project_file in project_files:
        print(f"Processing file: {project_file}")
        try:
            function(project_file, *args, **kwargs)
        except TypeError as e:
            logger.error(f"Function signature mismatch for file {project_file}: {e}")
        except Exception as e:
            logger.error(f"Error processing file {project_file}: {e}")
            logger.exception(e)
