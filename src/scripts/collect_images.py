import shutil
import os
import sys

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
drive_path = os.getenv("DRIVE_PATH")
metadata_path = os.path.join(drive_path, "Figures", "ssusi_metadata.csv")
metadata = pd.read_csv(metadata_path)

sample = metadata.sample(n=2500, replace=False, ignore_index=True)

def copy_image_files(image_filenames, source_dir, dest_dir):
    """Copies all the images in image_filenames from the source dir to dest dirf

    Args:
        image_filenames (list[str]): List of filenames to copy
        source_dir (str): source directory containing the images, or subdirectory with images
        dest_dir (str): desination directory to copy the images to
    """

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate over each filename in the DataFrame and copy the corresponding file
    for filename in image_filenames:
        year = filename[10:14]
        month = filename[14:16]
        day = filename[16:18]
        date = f"{year}-{month}-{day}"
        src_file = os.path.join(source_dir, date, f"{filename}.png")
        dest_file = os.path.join(dest_dir, f"{filename}.png")

        # Copy the file from source to destination
        shutil.copyfile(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")


for i in range(10):
    sliced = sample.iloc[i * 250 : (i + 1) * 250]
    image_filenames = sliced["Filename"].tolist()
    source_dir = os.path.join(drive_path, "Figures")
    dest_dir = os.path.join(drive_path, "Collected", f"Sample_{i+1}")
    copy_image_files(image_filenames, source_dir, dest_dir)
