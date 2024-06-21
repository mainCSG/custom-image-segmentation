import os
import shutil
import zipfile
import subprocess
from pathlib import Path
import argparse
from typing import Literal

def download_file(url: str, dest: Path) -> None:
    """
    Downloads a file from a given URL to a specified destination using wget.
    """
    subprocess.run(["wget", url, "-P", str(dest.parent), "-nc"], check=True)

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extracts a zip file to a specified directory.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_and_extract(version: Literal["lite", "v2", "both"], tmp_dir: Path) -> None:
    """
    Downloads and extracts QFlow data based on the specified version.
    """
    # Check if tmp directory exists
    if tmp_dir.exists():
        print(f"Temporary directory '{tmp_dir}' already exists. Skipping download.")
    else:
        # Create tmp directory
        tmp_dir.mkdir()

        # Download files
        if version in ("lite", "both"):
            lite_zip_path = tmp_dir / "data_qflow_lite.zip"
            lite_url = "https://data.nist.gov/od/ds/66492819760D3FF6E05324570681BA721894/data_qflow_lite.zip"
            print(f"Downloading {lite_url} to {lite_zip_path}")
            download_file(lite_url, lite_zip_path)

        if version in ("v2", "both"):
            v2_zip_path = tmp_dir / "data_qflow_v2.zip"
            v2_url = "https://data.nist.gov/od/ds/66492819760D3FF6E05324570681BA721894/data_qflow_v2.zip"
            print(f"Downloading {v2_url} to {v2_zip_path}")
            download_file(v2_url, v2_zip_path)

        print("Download complete.")

    # Extract files
    for zip_file in tmp_dir.glob("*.zip"):
        extract_zip(zip_file, tmp_dir)
        print(f"Extracted {zip_file.name} to {tmp_dir}")
        unzipped_folder = tmp_dir.joinpath(zip_file.stem)
        # Move .npy to data/raw
        dest_folder = tmp_dir / "raw" 
        if not dest_folder.exists():
            dest_folder.mkdir()

        if zip_file.stem == "data_qflow_v2":
            experimental_npy_folder = unzipped_folder / "experimental" / "exp_large"
            noiseless_hdf5_file = unzipped_folder / "simulated" / "noiseless_data.hdf5"
            print(f"Moving all files in {experimental_npy_folder} to {dest_folder}")
            for each_file in experimental_npy_folder.glob('*.*'): # grabs all files
                each_file.rename(dest_folder.joinpath(each_file.name)) # moves to parent folder.
            print(f"Moving simulated file in {noiseless_hdf5_file.parent} to {dest_folder}")
            noiseless_hdf5_file.rename(dest_folder.joinpath(noiseless_hdf5_file.name)) # moves to parent folder.
        else:
            print(f"Moving all files in {unzipped_folder} to {dest_folder}")
            for each_file in unzipped_folder.glob('*.*'): # grabs all files
                each_file.rename(dest_folder.joinpath(each_file.name)) # moves to parent folder.

        # Delete unzipped folder
        shutil.rmtree(unzipped_folder)

def main() -> None:
    """
    Main function to handle argument parsing and orchestrate the download and extraction process.
    """
    parser = argparse.ArgumentParser(description="Download QFlow data")
    parser.add_argument(
        "--version", 
        choices=["lite", "v2", "both"], 
        default="both", 
        help="Which version of the data to download (default: both)"
    )

    args = parser.parse_args()
    version = args.version

    # Set data directory
    data_dir = Path("data")
    # Check if tmp directory exists
    if not data_dir.exists():
        data_dir.mkdir()

    tmp_dir = Path("data/tmp")

    download_and_extract(version, tmp_dir)


if __name__ == "__main__":
    main()
