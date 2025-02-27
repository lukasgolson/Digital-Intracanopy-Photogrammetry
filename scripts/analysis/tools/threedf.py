import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from autokit import ToolConfig, PlatformData, ExternalTool
from autokit.ExecutableType import ExecutableType

from scripts.analysis.processing.scope_timer import ScopeTimer

logger = logging.getLogger(__name__)

class ThreeDF(ExternalTool):
    def __init__(self, base_dir: str = "./third-party", progress_bar: bool = True, lazy_setup: bool = False):
        super().__init__(Path(base_dir), progress_bar, lazy_setup)

    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            tool_name="three-df",
            platform_data={
                "windows": PlatformData(
                    url="https://github.com/3DFin/3DFin/releases/download/v0.4.0a3/3DFin.exe",
                    executable=Path("3DFin.exe"),
                    subdir=None
                ),
            },
            executable_type=ExecutableType.EXECUTABLE
        )

    def run(self, input_file: Path, output_path: Path, params_file: Path | None = None,
            normalize: bool = True,
            denoise: bool = True) -> tuple[Any, Any]:

        input_file = Path(input_file).resolve()
        output_path = Path(output_path).resolve()
        params_file = Path(params_file) if params_file is not None else None

        logger.info(str(output_path))
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        output_file_name = input_file.stem + "_dbh_and_heights.txt"
        output_file = output_path / output_file_name

        if output_file.exists():
            logger.info(f"Output file {output_file} already exists. Skipping processing and returning existing value.")
            return self.extract_dbh_from_file(output_file)

        if params_file is not None:
            params = f"config.ini"
        else:
            params = ""

        if not input_file.exists():
            raise FileNotFoundError(f"Input file {input_file} does not exist")

        with ScopeTimer("3DFIn"):
            os.environ['PYTHONIOENCODING'] = 'utf-8'

            status = self.run_command_cmd(
                f'cli "{input_file}" "{output_path}" {params} {"--normalize" if normalize else ""} {"--denoise" if denoise else ""} --export_txt',
                working_directory=self.calculate_dir())

            if status != 0:
                raise Exception(f"3DFIn failed with status {status}")

            dbh, height = self.extract_dbh_from_file(output_file)
            return dbh, height

    def extract_dbh_from_file(self, file_path: Path) -> (float, float):
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Input file {file_path} does not exist")

        df = pd.read_csv(file_path, delim_whitespace=True, header=None)

        # add column names
        df.columns = ['height', 'dbh', 'x', 'y']

        # get the first value of dbh and height
        dbh = df['dbh'].iloc[0] * 100
        height = df['height'].iloc[0]

        return dbh, height
