from pathlib import Path

from autokit import ExternalTool, ToolConfig, PlatformData
from autokit.ExecutableType import ExecutableType

from scripts.analysis.processing.scope_timer import ScopeTimer

class Delineation(ExternalTool):
    def __init__(self, base_dir: str = "./third-party", progress_bar: bool = True, lazy_setup: bool = False):
        super().__init__(base_dir, progress_bar, lazy_setup)

    # We call this "Tree Delineation", but it is actually only used for ground-identification (cloth)
    # CHM, normalization, and de-duplicating points; Tree identification is also done in the script but is not used
    # in further analysis.
    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            tool_name="LidR Delineation",
            platform_data={
                "windows": PlatformData(
                    url="https://github.com/lukasgolson/TreeDelineation/archive/refs/heads/main.zip",
                    subdir=Path("TreeDelineation-main"),
                    executable=Path("main.R")
                ),
            },
            executable_type=ExecutableType.RScript
        )

    def run(self, input_path: Path, output_path: Path):
        input_path = Path(input_path)
        output_path = Path(output_path)

        # ensure the input file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_path} does not exist")

        # ensure the output file does not exist
        if output_path.exists():
            output_path.unlink()

        with ScopeTimer("Delineation"):
            self.run_command(f'"{input_path.resolve()}" "{output_path.resolve()}"', working_directory=self.calculate_dir())

