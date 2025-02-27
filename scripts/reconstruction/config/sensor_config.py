import json
from pathlib import Path

# We have the new dyncfg module, but this code was written before we created it.
# This JSON file is more dynamic than the dyncfg module (using ini files), but it is also more complex.
# Until we can upgrade the dyncfg module to handle JSON files / grouped and arbitrary length structures, we will use this code.

class SensorConfig:
    def __init__(self, config_dir):
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "sensor.json"

    def read_config(self):
        # Default configuration with updated fiducial section
        default_config = {
            "sensor": {

                "shutter": "regularized",  # options are "disabled", "regularized", "full"
                "type": "frame",  # Frame, Fisheye, Spherical, Cylindrical, RPC

                "load_calibration_file": False,
                "calibration_file": "calibration.xml",

                "enable_sensor_correction": False,
                "pixel_height_mm": 0.0019,  # default values for the DJI Avata 1; values for FPV would be 0.00155
                "pixel_width_mm": 0.0019,

                # Values for height and width are in mm
                # DJI Avata 1: 0.0019 mm
                # DJI FPV: 0.00155 mm
            }
        }

        # Ensure the config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_file.exists():
            self.config_file.write_text(json.dumps(default_config, indent=4))
            self.config = default_config
        else:
            self.config = json.loads(self.config_file.read_text())

    def get_section(self, section_name):
        """Retrieve a specific section of the config."""
        return self.config.get(section_name, {})

    def save_config(self):
        """Save the current configuration to the file."""
        self.config_file.write_text(json.dumps(self.config, indent=4))
