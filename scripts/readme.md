# Drone-based Intracanopy Photogrammetry Reconstruction workflow

This workflow automates the process of 3D point cloud reconstruction for trees from first-person view drone imagery. 

It is based on the methodology described in the following paper.  Please cite the following when using it in your research or publications.
```text
Olson, L., Coops, N., Moreau, G., Hamelin, R., & Achim, A. (2025). The Assessment of Individual Tree Canopies Using Drone-Based Intra-Canopy Photogrammetry.
```
---

## Prerequisites

* **Python** 3.11 (required)
* **R version 4.3.1** (optional): Required for the `lidr` package used during feature extraction. If you do not use the analysis workflow, R is not required.
* **Operating System:** Primarily developed and tested on Microsoft Windows 10.  Due to Windows-specific dependencies, we cannot support compatibility with other operating systems.
* **Agisoft Metashape Professional:** A valid license for Agisoft Metashape Professional is **required** for the reconstruction component of this workflow.  The workflow will not function without it.

---
## Installation

### 1. Using the Pre-built Executable (Recommended)

The simplest way to use this workflow is by downloading the pre-built executable from the [Releases](link-to-releases) page.  No further installation is required; run the executable.

### 2. Manual Installation (Advanced)

If you wish to install the workflow manually (e.g., for development or customization), follow these steps:

1. **Create a Virtual Environment:** Using a virtual environment, like `venv`, to manage dependencies is highly recommended:

   ```bash
   python3.11 -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```
   
2. **Install Dependencies:** Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```
   
3. **Install remaining dependencies:** The workflow requires Agisoft Metashape Python API to be installed in your environment. For guided-installation, run:

    ```bash
    python main.py setup
    ```

---
## Usage

### Displaying Command Help

```bash
main.py --help
```

### Installing Dependencies

Before using the workflow, you will need to install the required dependencies. To do this, you can run the following command:

```bash
python main.py setup [options]
```

  - `--requirements [true/false]`: Install Python dependencies from `requirements.txt`. This is enabled by default.
  - `--metashape_tos [true/false]`: Accept the Metashape API terms of service to guide Agisoft Metashape installation. This is disabled by default.

### Reconstruction

To run the reconstruction workflow, run the following command:

```bash
python main.py reconstruct process [options]
```

  - `config_path`: Path to the configuration file. The default is `photogrammetry-config.ini.`

#### Usage

A configuration file will be created on the first run. This file can be edited to customize the reconstruction process.
Default values are appropriate for most use cases.

Create a subdirectory in the configured input directory (default: `data/input`) and place it inside your video files.
The workflow will process all video files in the input directory and output the results to the output directory.
(default: `data/export`).

### Clip

To clip each point cloud to the inside of a convex hull formed by the drone flight, run the following command:

```bash
python main.py analysis clip CLOUD_PATH [EXPORT_PATH] [QUANTILE] [BUFFER_PERCENT]
```

  - `CLOUD_PATH`: Path to the point cloud file. The workflow will look for the point cloud in the `data/export` directory by default.
  - `EXPORT_PATH`: Path to the output directory. The clipped point cloud will be saved in the `data/export` directory by default.
  - `QUANTILE`: The quantile value for the convex hull. Default 0.
  - `BUFFER_PERCENT`: The percentage of the convex hull to buffer. Default 0.1 (10%).

### Analysis

To run the analysis workflow, run the following command:

```bash
python main.py analysis predict [options]
```

  - `config_path`: Path to the configuration file. The default is `predict-config.ini`.

#### Usage

A configuration file will be created on the first run. This file can be edited to customize the analysis process.

In the configured input directory, create a sub-directory for each site. Place the point cloud files inside each site directory containing a tree to be analyzed.

-----

## License

This project is licensed under the **MIT License**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

You are free to use, modify, and distribute this software for commercial and non-commercial purposes as long as you comply with the terms of the MIT License.  See the [LICENSE](link-to-your-license-file) file for the full text of the license.

**In short, the MIT License allows you to:**

  * ✓ **Freely use the software:** For any purpose, commercial or private.
  * ✓ **Modify the software:** Adapt it to your needs.
  * ✓ **Distribute the software:** Share it with others.
  * ✓ **Use it privately.**
  * ✓ **Use it commercially.**

**Under the condition that you:**

  * ✗  **Include the original MIT license and copyright notice** in all copies or substantial portions of the software.
  * ✗  **The software is provided "as is" without warranty.** The authors or copyright holders are not liable for any damages or issues caused by using this software.

> [!WARNING]
> While this project's source code is licensed under the MIT License, it relies on third-party dependencies that may be licensed under different terms.  **It is your responsibility to review the licenses of all dependencies** listed in `requirements.txt` and any other dependency documentation to ensure compliance with their respective terms. The MIT License applies solely to the code developed in this repository.


-----

## Contributing

We welcome contributions to this software; your input is valuable. If you find a bug, have a feature request, or want to contribute code improvements.

**Here are some ways you can contribute:**

  * **Report Bugs:** If you encounter any issues or unexpected behaviour, please open a new issue on the [Issues tab](link-to-your-github-issues-tab). Please provide detailed steps to reproduce the bug, your operating system, and any relevant error messages.
  * **Suggest Enhancements:** Do you have an idea for a new feature or improvement to the workflow? Feel free to submit a feature request as an issue on the [Issues tab](link-to-your-github-issues-tab).
  * **Code Contributions (Pull Requests):** If you are a developer and want to contribute code, we encourage you to fork the repository, create a branch for your changes, and submit a pull request (PR). Please ensure your code adheres to the project's coding style and includes appropriate tests if applicable.

**Before contributing code, please:**

1.  **Check existing issues and pull requests:** To avoid duplicate effort, please check if a similar issue or PR exists.
2.  **Discuss your proposed changes:** For significant changes or new features, it is recommended to open an issue first to discuss your approach with the maintainers before investing significant development time.

-----

## Support

Please use the **[Issues tab](https://github.com/lukasgolson/Digital-Intracanopy-Photogrammetry/issues)** on this GitHub repository for questions, issues, or general support requests.

**When submitting an issue, please be as detailed as possible:**

  * **Describe your issue clearly and concisely.**
  * **Provide steps to reproduce the issue (if applicable).**
  * **Include your operating system and Python version.**
  * **Attach any relevant error messages, configuration files, or input data snippets (if appropriate).**

**Please avoid, when possible, using email or other private channels for support requests related to the workflow's functionality or bugs. Using the Issues tab allows us to track issues effectively and makes solutions available to the wider community.**