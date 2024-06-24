from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image

def convert_array_to_image(raw_data: np.ndarray, filepath: str) -> None:
    # Create JPG
    if not isinstance(raw_data, np.ndarray):
        return
    
    raw_data_normalized = ((raw_data - raw_data.min()) / (raw_data.max() - raw_data.min()))
    log_data = np.log1p(raw_data_normalized)
    # normalized_raw_data = raw_data / np.amax(raw_data)
    image = Image.fromarray(
        (255 * log_data).astype(np.uint8)
    )
    image.save(filepath.parent / f"{filepath.stem}.jpg")

def extract_raw_data(npy_filepath: str, data_type: str) -> list:
    file_data = np.load(
        npy_filepath, allow_pickle=True
    ).item()

    voltage_P1_key = "x" if "exp" in npy_filepath.stem else "V_P1_vec"
    voltage_P2_key = "y" if "exp" in npy_filepath.stem else "V_P2_vec"
    V_P1 = np.array(file_data[voltage_P1_key])
    V_P2 = np.array(file_data[voltage_P2_key])
    N = len(V_P1)
    M = len(V_P2)

    try:
        file_output = file_data['output']
    except KeyError:
        file_output = file_data
    
    if isinstance(file_output, list):

        raw_data = np.array(
            [np.average(pixel[data_type]) for pixel in file_output]
        ).reshape((N,M))

    if isinstance(file_output, dict):

        try:
            raw_data = np.array(file_output[data_type])
        except KeyError:
            raw_data = -1

    return raw_data, V_P1, V_P2

def main(data_type: str = None):

    data_dir = Path("data")

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    all_dirs = [train_dir, val_dir, test_dir]

    for dir in all_dirs:
        for file in os.listdir(dir):
            if file.endswith(".npy"):
                print(f"Converting {file} to .jpg")
                raw_data, _, _ = extract_raw_data(npy_filepath = dir / file, data_type=data_type)
                convert_array_to_image(raw_data, dir / file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy files to jpg")
    parser.add_argument(
        "--data_type", 
        choices=["current", "sensor"], 
        default="sensor", 
        help="Which data type do you want to generate (default: sensor)"
    )
    args = parser.parse_args()

    main(data_type = args.data_type)
    
    
