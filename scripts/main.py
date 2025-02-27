# setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import os
import sys
from pathlib import Path

from scripts.Utilities.project_version import ProjectVersion

from scripts.Utilities.numba_redirect import numba_redirect

numba_redirect()

minimal_mode = False

# Get the directory of your main script
main_dir = os.path.dirname(os.path.abspath(__file__))

# Add the subfolder to the Python path
analysis_dir = os.path.join(main_dir, "analysis")
reconstruction_dir = os.path.join(main_dir, "reconstruction")

sys.path.append(analysis_dir)
sys.path.append(reconstruction_dir)

try:
    import click
    from loguru import logger
    from analysis.analysis_cli import analysis_cli
    from reconstruction.reconstruction_cli import reconstruction_cli, install_command
except ImportError as e:
    minimal_mode = True
    print("The application cannot start because required Python packages are missing: " + e.msg)

if minimal_mode:
    logger = logging.getLogger(__name__)

else:
    @click.group()
    def cli():
        pass


    cli.add_command(reconstruction_cli)
    cli.add_command(analysis_cli)

    cli.add_command(install_command)

    logger.remove()

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )


def log_header():
    local_dir = Path(__file__).parent
    workflow_hash = ProjectVersion(local_dir).hash(1e-9)
    logger.info(f"Intracanopy Photogrammetry Workflow Version: {workflow_hash}")

    # check if MetaShape is installed
    import scripts.Utilities.check_metashape_status as check_metashape_status
    installed, licensed = check_metashape_status.check_agisoft_license()
    if not installed:
        logger.error(f"Agisoft Metashape API is not installed or a system license is not installed. Please run setup for guided installation if you haven't already.")
    if installed and not licensed:
        logger.warning(
            f"Agisoft Metashape API is installed, but a system license was not found. Please ensure a valid license is installed. Note: If you have configured your license in the config file, this warning can be ignored.")



def parse_arguments():
    import argparse
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="This application requires certain Python packages that are currently missing. The following commands are available:"

    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'setup' command parser
    setup_parser = subparsers.add_parser(
        "setup", help="Download and install required packages including the Agisoft Metashape API."
    )
    setup_parser.add_argument(
        "install_metashape",
        nargs="?",
        default="true",
        help="Download & install Agisoft Metashape API. Accepts true/false or yes/no (default: true)."
    )

    setup_parser.add_argument(
        "metashape_tos",
        nargs="?",
        default="false",
        help="Agree to the Agisoft Metashape API Terms of Service before installation. Accepts true/false or yes/no (default: false).")

    setup_parser.add_argument(
        "bin_path",
        nargs="?",
        default="bin",
        help="Path to the directory where binary files will be stored (default: 'bin')."
    )

    args = parser.parse_args(args=None if sys.argv[1:] else ['setup', '--help'])

    return args, parser


if __name__ == "__main__":
    cli_command = " ".join(sys.argv)

    log_header()

    if not minimal_mode:

        cli()
    else:
        logger.error("The application cannot start because required Python packages are missing.")

        if (len(sys.argv) > 1 and sys.argv[1] != "setup") or len(sys.argv) == 1:
            logger.error(
                "Please install the required dependencies by following the setup command usage instructions below, "
                "or manually using 'pip install -r requirements.txt' or equivalent.")
        logger.error(
            f"Once the packages are installed, please run '{cli_command}' again. For more information, see the README file.")
        args, parser = parse_arguments()

        # Check the provided command and act accordingly.
        if args.command == "setup":
            logger.info("Installing Packages")

            from reconstruction.utils.install_packages import install_requirements, download_wheels, install_wheels

            if args.install_metashape.lower() in ("false", "no"):
                install_metashape = False
            else:
                install_metashape = True

            if args.metashape_tos.lower() in ("false", "no"):
                accept_ms_tos = False
            else:
                accept_ms_tos = True

            bin_path = Path(args.bin_path)

            download_wheels(bin_path, install_metashape, accept_ms_tos)
            install_requirements(bin_path)
            install_wheels(bin_path)

            logger.info("Installation Complete")

        else:
            logger.error(f"Invalid Command. Please run 'python {sys.argv[0]} help' for usage information.")
