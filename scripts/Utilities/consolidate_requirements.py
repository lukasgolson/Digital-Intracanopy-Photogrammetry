#!/usr/bin/env python3
import os
import logging
from typing import Dict, List, Optional, Tuple

chardet_installed = False
packaging_installed = False

try:
    import chardet  # Install via: pip install chardet

    chardet_installed = True
except ImportError:
    pass

try:
    from packaging.requirements import Requirement
    from packaging.version import Version, InvalidVersion
    packaging_installed = True
except ImportError:
    pass

if not chardet_installed and not packaging_installed:

    raise ImportError("Please install the required packages: chardet, packaging: pip install chardet packaging")
elif not chardet_installed:
    raise ImportError("Please install the required package: chardet: pip install chardet")
elif not packaging_installed:
    raise ImportError("Please install the required package: packaging: pip install packaging")

# Configure logging with increased verbosity.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
logger.addHandler(handler)


def detect_encoding(file_path: str) -> str:
    """
    Detects the encoding of a file using chardet.
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    detection = chardet.detect(raw_data)
    encoding = detection.get('encoding') or 'utf-8'
    confidence = detection.get('confidence')
    logger.debug(f"File '{file_path}' raw encoding detection: {detection}")
    logger.info(f"Detected encoding for {file_path}: {encoding} (Confidence: {confidence})")
    return encoding


def read_file(file_path: str) -> Optional[str]:
    """
    Reads a file using its detected encoding.
    """
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        logger.debug(f"Read {len(content)} characters from file {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def parse_requirements(content: str, file_path: str) -> List[str]:
    """
    Extracts valid requirement lines from file content.
    """
    reqs = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            Requirement(line)
            reqs.append(line)
        except Exception as e:
            logger.warning(f"Could not parse line '{line}' in {file_path}: {e}")
    logger.debug(f"Parsed {len(reqs)} valid requirement lines from {file_path}")
    return reqs


def extract_version(req: Requirement) -> Optional[Version]:
    """
    Extracts the version from a Requirement if it has a pinned version specifier.
    Checks for both '==' and '~=' operators.

    Returns:
        A Version object if found, otherwise None.
    """
    for spec in req.specifier:
        if spec.operator in ("==", "~="):
            try:
                version_obj = Version(spec.version)
                logger.debug(f"Extracted version {version_obj} from requirement {req}")
                return version_obj
            except InvalidVersion:
                logger.warning(f"Invalid version format for package {req.name}: {spec.version}")
                return None
    logger.debug(f"No valid version specifier found in requirement {req}")
    return None


def consolidate_package_requirements(
        package: str, entries: List[Tuple[str, Optional[Version]]]
) -> List[str]:
    """
    Consolidates multiple requirement entries for a single package.

    - If entries have different major versions, all entries are included and a warning is logged.
    - If entries differ only in minor/patch versions (i.e. same major version), only the highest version is kept.
    - Entries without version info are kept as is.

    Returns:
        A list of consolidated requirement lines for the package.
    """
    # Group entries by major version; use None for entries with no version info.
    groups: Dict[Optional[int], List[Tuple[str, Optional[Version]]]] = {}
    for line, version in entries:
        group_key = version.major if version is not None else None
        groups.setdefault(group_key, []).append((line, version))
    logger.debug(f"Package '{package}' version groups: {[(k, [e[0] for e in v]) for k, v in groups.items()]}")

    consolidated_lines: List[str] = []
    if len(groups) > 1:
        # Multiple major versions are present.
        for group_key, group_entries in groups.items():
            if group_key is not None:
                best_entry = max(group_entries, key=lambda x: x[1])
                consolidated_lines.append(best_entry[0])
                logger.debug(
                    f"For package '{package}', major version {group_key}: selected best entry '{best_entry[0]}'")
            else:
                unique = set(line for line, _ in group_entries)
                consolidated_lines.extend(unique)
                logger.debug(f"For package '{package}', no version info: added entries {unique}")
        logger.warning(
            f"Different major versions found for package '{package}': {list(groups.keys())}. "
            f"Including all entries: {consolidated_lines}"
        )
    else:
        # Only one major version (or no version info) exists.
        group_entries = next(iter(groups.values()))
        if len(group_entries) > 1:
            logger.warning(f"Multiple versions for '{package}': {[(line, version) for line, version in group_entries]}")

        if next(iter(groups)) is not None:
            best_entry = max(group_entries, key=lambda x: x[1])
            consolidated_lines.append(best_entry[0])
            logger.debug(f"For package '{package}', single major version group: selected best entry '{best_entry[0]}'")
        else:
            unique = set(line for line, _ in group_entries)
            consolidated_lines.extend(unique)
            logger.debug(f"For package '{package}', no version info: added entries {unique}")

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(consolidated_lines))


def consolidate_requirements(root_dir: str) -> None:
    """
    Consolidates all non-root requirements.txt files from the directory tree.

    Parses each requirement, groups them by package, and consolidates version
    specifications based on the following rules:
      - If different major versions exist for the same package, output all versions.
      - If only minor/patch differences exist, output only the highest version.

    The final consolidated, sorted list of requirements is written to a single
    requirements.txt in the root directory.
    """
    output_file = os.path.join(root_dir, 'requirements.txt')
    requirements_by_package: Dict[str, List[Tuple[str, Optional[Version]]]] = {}
    file_count = 0
    total_req_count = 0

    for subdir, _, files in os.walk(root_dir):
        # Skip the root directory to avoid processing the consolidated file.
        if os.path.abspath(subdir) == os.path.abspath(root_dir):
            continue
        for file in files:
            if file == 'requirements.txt':
                file_path = os.path.join(subdir, file)
                logger.info(f"Processing file: {file_path}")
                content = read_file(file_path)
                if content is None:
                    logger.error(f"Could not decode file {file_path}.")
                    continue
                file_count += 1
                req_lines = parse_requirements(content, file_path)
                total_req_count += len(req_lines)
                for line in req_lines:
                    try:
                        req = Requirement(line)
                    except Exception as e:
                        logger.warning(f"Could not parse requirement line '{line}' in {file_path}: {e}")
                        continue
                    pkg_name = req.name.lower()
                    version = extract_version(req)
                    requirements_by_package.setdefault(pkg_name, []).append((line, version))
                    logger.debug(f"Added requirement for package '{pkg_name}': '{line}' with version {version}")

    logger.info(f"Processed {file_count} files with a total of {total_req_count} requirement lines.")

    consolidated_lines: List[str] = []
    for pkg in sorted(requirements_by_package.keys()):
        pkg_lines = consolidate_package_requirements(pkg, requirements_by_package[pkg])
        consolidated_lines.extend(pkg_lines)
        logger.debug(f"Consolidated requirement for package '{pkg}': {pkg_lines}")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in consolidated_lines:
            outfile.write(line + "\n")
    logger.info(f"Consolidated requirements written to {output_file}")
    logger.debug(f"Final consolidated requirements: {consolidated_lines}")


def main() -> None:
    """
    Main entry point:
      - Removes any existing consolidated requirements.txt in the root.
      - Initiates the consolidation process.
    """
    root_directory = os.getcwd()
    output_path = os.path.join(root_directory, 'requirements.txt')
    if os.path.exists(output_path):
        os.remove(output_path)
        logger.debug(f"Removed existing consolidated requirements file at {output_path}")
    consolidate_requirements(root_directory)


if __name__ == "__main__":
    main()
