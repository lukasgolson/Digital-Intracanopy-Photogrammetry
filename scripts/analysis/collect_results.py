import json
import os
import re
from pathlib import Path
from typing import List

import pandas as pd
from alive_progress import alive_bar
from pandas import DataFrame



def extract_name(path):
    """Extract the folder name and tree name from the given path."""
    initial_path = Path("data/output")

    path = Path(path)

    relative_path = path.relative_to(initial_path)
    parts = relative_path.parts
    site_name = parts[0]
    tree = parts[1]
    return site_name, tree


def clean_dataframe(df: DataFrame) -> DataFrame:
    df["Tree"] = df["Tree"].str.replace(r"tree", "", case=False, regex=True)
    df["Site"] = df["Site"].str.replace(r"site", "", case=False, regex=True)
    df = df.drop(columns=["Input"])
    df.columns = [col.capitalize() for col in df.columns]
    df.columns = [
        re.sub(r"dbh", "DBH", col, flags=re.IGNORECASE) for col in df.columns
    ]
    return df


def collect_files(root_dir: str = "data/output", extension=".csv", callback=None) -> List[str]:
    collected_files: List[str] = []



    for root, _, files in os.walk(root_dir):

        # skip any directory with debug in the name
        if "debug" in root:
            continue

        for file in files:

            if file.endswith(extension):
                collected_files.append(os.path.join(root, file))

                if callback is not None:
                    callback()

    return collected_files




def write_record(path: Path, first_file: bool, output_file: str):
    """Process an individual file, clean data, and append it to the output file."""
    site_name, tree = extract_name(path)
    data = pd.read_csv(path)

    print(f"Processing {site_name}/{tree}")
    data["Tree"] = tree
    data["Site"] = site_name

    # Remove any columns with "DBH." in their name
    data = data.loc[:, ~data.columns.str.contains("DBH.", case=False)]

    # Clean the dataframe
    data = clean_dataframe(data)

    # Write to the output file
    with open(output_file, mode="a", newline="", encoding="utf-8") as output:
        data.to_csv(output, index=False, header=first_file)  # Write headers only for the first file



def combine_results(input_dir: str = "data/output", output_file: str = "data/results.csv", downsampling_output_file: str = "data/results_cleaning.csv"):
    """Combine all CSV files in the given directory into a single file."""

    csv_file_names = collect_files(input_dir, ".csv")

    json_file_names = collect_files(input_dir, ".json")



    if os.path.exists(output_file):
        os.remove(output_file)

    if os.path.exists(downsampling_output_file):
        os.remove(downsampling_output_file)

    # Process each file
    first_csv_file = True
    for path in csv_file_names:
        write_record(Path(path), first_csv_file, output_file)
        first_csv_file = False



    first_json_file = True  # Write header only for the first JSON file

    for path in json_file_names:
        # Skip empty files
        if not os.path.exists(path) or os.stat(path).st_size == 0:
            print(f"Skipping empty or missing file: {path}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {path}")
            continue

        # Convert dictionary to DataFrame with one row
        data = pd.DataFrame([raw_data])

        # Add metadata columns
        site_name, tree = extract_name(path)
        data["Tree"] = tree
        data["Site"] = site_name

        # Append to CSV
        with open(downsampling_output_file, mode="a", newline="", encoding="utf-8") as output:
            data.to_csv(output, index=False, header=first_json_file)

        first_json_file = False  # Ensure header is written only once


if __name__ == "__main__":
    combine_results()
