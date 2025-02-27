import json
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List
from urllib.parse import urlparse


import logging

from scripts.Utilities.dir_helpers import get_all_files

logger = logging.getLogger(__name__)


# when using Windows Sandbox, the default SSL context does not work.
# This function will use the certifi package to get a valid SSL context.
def get_ssl_context():
    try:
        import certifi
        ca_bundle_path = certifi.where()
        logger.info(f"Using certifi CA bundle: {ca_bundle_path}")
        return ssl.create_default_context(cafile=ca_bundle_path)
    except ImportError:
        logger.info("certifi is not installed; using default SSL context")
        return ssl.create_default_context()


def run_pip(bin_path: Path, command: List[str]):

    bin_path = Path(bin_path)

    if not (bin_path / 'pip.pyz').exists():
        raise FileNotFoundError(f"pip.pyz not found in {bin_path}")
    subprocess.check_call([sys.executable, bin_path / 'pip.pyz'] + command)


def install_requirements(bin_path: Path):
    """
    Installs the requirements from the requirements.txt file.
    :return:
    """
    run_pip(bin_path, ['install', '-U', 'pip', 'setuptools', 'wheel'])
    run_pip(bin_path, ['install', '-r', 'requirements.txt'])


def install_package(bin_path, package, path_to_whl=None):
    """
    Installs the specified package.
    :param package:
    :param path_to_whl:
    :return:
    """
    try:
        globals()[package] = __import__(package)
    except ImportError:
        run_pip(bin_path, ['install', path_to_whl] if path_to_whl else ['install', package])


def download_file(url: str, output_path: Path):
    ssl_context = get_ssl_context()

    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
    urllib.request.install_opener(opener)

    try:
        urllib.request.urlretrieve(url, output_path)
    except urllib.error.URLError as e:
        # If we get a SSL error, we can advise the user to install certifi

        if isinstance(e.reason, ssl.SSLError):
            logger.warning("SSL error occurred. You may need to install the certifi package to resolve this issue.")
            logger.warning("Run 'pip install certifi' in your environment to install the package.")


def download_wheels(bin_path: Path, download_metashape: bool = True, accept_ms_tos: bool = False):
    bin_path = Path(bin_path)

    Path.mkdir(bin_path, exist_ok=True)

    pip_url = "https://bootstrap.pypa.io/pip/pip.pyz"
    metashape_url = "https://storage.yandexcloud.net/download.agisoft.com/Metashape-2.1.2-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl"

    urls_file = Path("setup-urls.json")

    if urls_file.exists():
        with open(urls_file, 'r') as file:
            urls = json.load(file)
            pip_url = urls.get("pipURL")
            metashape_url = urls.get("metashapeURL")
    else:
        logger.error(
            f"File {urls_file} not found. Please create a file named setup-urls.json with the following content. Make sure to validate and ensure that the URLs are safe and up-to-date.")
        logger.error(json.dumps({"pipURL":       pip_url,
                                 "metashapeURL": metashape_url}))

    download_file(pip_url, bin_path / "pip.pyz")

    if download_metashape:

        if not accept_ms_tos:
            logger.info("Opening the Agisoft Metashape download page and EULA for your information.")

            # press any key to continue
            press_any_key()


            # open the download page in the default browser
            import webbrowser
            webbrowser.open("https://www.agisoft.com/downloads/installer/")
            webbrowser.open("https://www.agisoft.com/pdf/metashape-pro_eula.pdf")

            # Add consent mechanism here
            consent = input(
                "By proceeding, you confirm that you have a valid Agisoft Metashape Professional license and agree to their Terms of Service. Type 'yes' to continue with the download and installation: ").lower()
        else:
            consent = 'yes'

        if consent == 'yes' or consent == 'y':
            logger.info("User consent granted. Proceeding with download.")
            logger.info(
                f"Downloading Metashape wheel from {metashape_url}")
            filename = urlparse(metashape_url).path.split('/')[-1]
            download_file(metashape_url, bin_path / filename)
        else:
            logger.info("User consent not granted. Download cancelled.")
            logger.info(
                "Download cancelled by user. Please ensure you have the necessary license and permissions before attempting to download again.")
            sys.exit(1)


def install_wheels(bin_path: Path):

    bin_path = Path(bin_path)

    Path.mkdir(bin_path, exist_ok=True)

    wheels = get_all_files(bin_path, '*.whl')
    for wheel in wheels:
        package_name = wheel.stem.split('-')[0]
        install_package(bin_path, str(package_name), bin_path / wheel.name)


def press_any_key():
    """Waits for the user to press any key before continuing."""
    print("Press any key to continue...", end="", flush=True)

    if sys.platform.startswith("win"):
        import msvcrt
        msvcrt.getch()  # Wait for a key press on Windows

if __name__ == "__main__":

    # configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    bins = "/bin"

    download_wheels(bins)

    install_requirements(bins)
    install_wheels(bins)
