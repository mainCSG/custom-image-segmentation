from pathlib import Path
import os
import argparse
import random
import shutil

def partition(files_list: list[str],
                train_ratio: float, 
                val_ratio: float,
                data_dir: str,
                train_dir: str,
                val_dir: str,
                test_dir: str,
                random: bool = False) -> None:
    
    if random:
        # Sets the random seed 
        random.seed(42)

        # Shuffle the list of image filenames
        random.shuffle(files_list)

    # determine the number of images for each set
    train_size = int(len(files_list) * train_ratio)
    val_size = int(len(files_list) * val_ratio)

    for i, each_file in enumerate(files_list): # grabs all files
        file_path = data_dir / each_file
        if i < train_size:
            trg_path = train_dir
        elif i < train_size + val_size:
            trg_path = val_dir
        else:
            trg_path = test_dir

        if not trg_path.exists():
            trg_path.mkdir()

        shutil.copy(file_path, trg_path)

def main(data_dir: Path, train_ratio: float, val_ratio: float, test_ratio: float, random_partition: bool) -> None:
    ratio = (train_ratio, val_ratio, test_ratio)
    assert sum(ratio) == 1, "Ratios need to sum to unity."
    print(f"Splitting data found in {data_dir} to {data_dir.parent.parent}")
    
    train_dir = data_dir.parent.parent / "train"
    val_dir = data_dir.parent.parent / "val"
    test_dir = data_dir.parent.parent / "test"

    # Define a list of file extensions
    file_extensions = ['.npy']

    # Create a list of image filenames in 'data_path'
    simulated_files_list = [
        filename for filename in os.listdir(data_dir) 
        if (os.path.splitext(filename)[-1] in file_extensions) and ("exp" not in filename)
        ]
    experimental_files_list = [
        filename for filename in os.listdir(data_dir) 
        if (os.path.splitext(filename)[-1] in file_extensions) and ("exp" in filename)
        ]
    
    # First copy experimental files over
    partition(
        experimental_files_list, 
        train_ratio, 
        val_ratio, 
        data_dir, 
        train_dir, 
        val_dir,
        test_dir,
        random=random_partition
    )
    partition(
        simulated_files_list, 
        train_ratio, 
        val_ratio, 
        data_dir, 
        train_dir, 
        val_dir, 
        test_dir,
        random=random_partition
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Partition QFlow data")
    parser.add_argument(
        "--train", 
        type=float,
        default=0.8, 
        help="Train ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val", 
        type=float,
        default=0.1, 
        help="Validation ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test", 
        type=float,
        default=0.1, 
        help="Test ratio (default: 0.1)"
    )
    parser.add_argument(
        "--random", 
        choices=["Y", "N", "y", "n"], 
        default="both", 
        default="N", 
        help="Randomly partition the data (default: N)"
    )

    args = parser.parse_args()

    # Set data directory
    data_dir = Path("data/tmp/raw")

    main(data_dir, args.train, args.val, args.test, random_partition=args.random)