import sys
from pathlib import Path

import click
from dyncfg import DynamicConfig
from loguru import logger

from scripts.Utilities.dir_helpers import execute_on_each_file
from scripts.Utilities.check_metashape_status import check_agisoft_license
from scripts.Utilities import interrupt_manager
from utils.install_packages import install_requirements, download_wheels, install_wheels

from commands.export import export
from commands.video_to_pc import process_videos


def validate_metashape(need_license: bool = True):
    is_installed, is_licensed = check_agisoft_license()

    if not is_installed:
        logger.error(
            f"Agisoft Metashape is not installed. If you are using the reconstruction pipeline, please run setup and try again.")

        sys.exit(1)

    if is_installed and need_license and not is_licensed:
        logger.warning(
            f"Agisoft Metashape is installed, but a Professional Edition license has not been activated on this machine or is missing from the 'agisoft_LICENSE' environment variable."
            f"Please activate a license and try again, ensure that a valid license key is provided in the configuration file, or activate Agisoft Metashape in a full application installation.")


@click.command("setup", help="Install the required packages.")
@click.argument("bin_path", type=click.Path(exists=False), default="bin")
@click.option("--requirements", is_flag=True, help="Install the requirements from the requirements.txt file. If disabled, only the Agisoft Metashape wheel will be installed.")
@click.option("--metashape_tos", is_flag=True, help="Agree to the Agisoft Metashape API Terms of Service before installation.")
def install_command(bin_path: str, requirements : bool, metashape_tos : bool):
    download_wheels(bin_path, True, metashape_tos)
    if requirements:
        install_requirements(bin_path)
    install_wheels(bin_path)

@click.command("process", help="Process video files to generate point clouds.")
@click.argument("config_file_path", type=click.Path(dir_okay=False, exists=False), default="photogrammetry-config.ini")
def process_video_command(config_file_path):

    # we do our own validation in the process_videos function

    process_videos(Path(config_file_path))


@click.command("export", help="Recursively export all Metashape projects in directory.")
@click.argument("config_file_path", type=click.Path(dir_okay=False, exists=False), default="photogrammetry-config.ini")
@click.argument("project_path", type=click.Path(exists=True))
@click.argument("export_path", type=click.Path(exists=False), default="export")
@click.argument("cameras_only", default=False)
@click.argument("rebuild", default=False)
def export_command(config_file_path, project_path, export_path, cameras_only, rebuild: bool):

    config = DynamicConfig(config_file_path)
    manage_license = config['Settings']['manage_metashape_license'].or_default("False").as_bool()
    metashape_license = config['Settings']['metashape_license'].or_default("").as_str()

    if manage_license:
        from scripts.reconstruction.tools.agisoft_metashape import activate_license
        activate_license(metashape_license)

    validate_metashape()

    suffices = ["*.psx"]

    if not Path(export_path).exists():
        Path(export_path).mkdir(parents=True)

    execute_on_each_file(Path(project_path), suffices, export, export_path, cameras_only, rebuild)

    if manage_license:
        from scripts.reconstruction.tools.agisoft_metashape import deactivate_license
        deactivate_license()





interrupt_manager.register_signal_handlers()


@click.group(name="reconstruct", help="All photogrammetry commands.")
def reconstruction_cli():
    pass

reconstruction_cli.add_command(process_video_command)
reconstruction_cli.add_command(export_command)

if __name__ == '__main__':

    try:
        # Remove default handler to avoid double logging
        logger.remove()

        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        )

        logger.add(
            "/logs/log_{time}.txt",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            enqueue=True,  # Enable message queuing
            rotation="50 MB",
            compression="zip"
        )

        reconstruction_cli()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
