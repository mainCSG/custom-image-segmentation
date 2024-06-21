import sys
import os
import numpy as np
import h5py
from pathlib import Path
from typing import Optional
import argparse

def convert_files(h5py_dir: Path, npy_dir: Path) -> None:
    """
    Convert HDF5 files to NumPy arrays.

    Args:
        h5py_dir (Path): Path to the folder containing the raw h5py QFlow data.
        npy_dir (Path): Path to the folder for converted npy QFlow data.

    Returns:
        None
    """
    raw_files = os.listdir(h5py_dir)

    # Process files in the raw folder
    for file in raw_files:
        if file.endswith(".hdf5"):
            h5py_file_path = h5py_dir / file
            # Save the processed results to the npy folder
            save_as_npy(h5py_file_path, npy_dir)
            
def save_as_npy(h5py_file_path: Path, npy_dir: Path) -> None:
    """
    Convert HDF5 file to NumPy array and save. This function is based on the
    QFlow v2 dataset_structure 

    Args:
        h5py_file_path (Path): Path to the HDF5 file.
        npy_dir (Path): Path to the folder for converted npy QFlow data.

    Returns:
        None
    """

    fileID = 0
    with h5py.File(h5py_file_path, "r") as f:
        
        d = [n for n in f.keys()]
        npy_dict = {}
        npy_dict['output'] = {}
        for data in d:
            
            qflow_data = f[data]

            npy_dict['V_P1_vec'] = np.array(qflow_data["V_P1_vec"])
            npy_dict['V_P2_vec'] = np.array(qflow_data["V_P2_vec"])
            npy_dict['output']['sensor'] = np.array(qflow_data['output']['sensor'])
            npy_dict['output']['state'] = np.array(qflow_data['output']['state'])

            np.save(npy_dir / f"{data}_file{fileID}.npy", npy_dict)

            fileID += 1

def main(h5py_dir: Path, npy_dir: Path) -> None:
    """
    Main function to convert HDF5 files to NumPy arrays.

    Args:
        h5py_dir (str, optional): Path to the folder containing the raw h5py QFlow data. 
        npy_dir (str, optional): Path to the folder for converted npy QFlow data.

    Returns:
        None
    """
    h5py_dir = Path(h5py_dir)
    npy_dir = Path(npy_dir)
    convert_files(h5py_dir, npy_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 files to NumPy arrays")
    parser.add_argument(
        "h5py_dir", 
        nargs="?", 
        default="./data/tmp/raw",
        help="Path to the folder containing the raw h5py QFlow data (default: ./data/tmp/raw)"
        )
    parser.add_argument(
        "npy_dir", 
        nargs="?", 
        default="./data/tmp/raw", 
        help="Path to the folder for converted npy QFlow data (default: ./data/tmp/raw)"
        )
    args = parser.parse_args()

    main(args.h5py_dir, args.npy_dir)
