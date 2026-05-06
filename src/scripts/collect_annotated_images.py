import os
import json
import shutil

ANNOTAIONS_PATH = "/mnt/e/Collected_non_labled/Sample_2/annotations/instances_default.json"
IMAGES_DIR = "/mnt/e/Collected_non_labled/Sample_2"
OUTPUT_DIR = "/mnt/e/Collected_non_labled/Sample_2_aggregated_images"

with open(ANNOTAIONS_PATH, "r") as f:
    annotations = json.load(f)
    
categories = annotations["categories"]
images = annotations["images"]

for category in categories:
    category_id = category["id"]
    category_name = category["name"]
    
    category_dir = os.path.join(OUTPUT_DIR, category_name)
    os.makedirs(category_dir, exist_ok=True)
    
    for annotation in annotations["annotations"]:
        if annotation["category_id"] == category_id:
            image_id = annotation["image_id"]
            image_names = [img["file_name"] for img in images if img["id"] == image_id]
            paths = [os.path.join(IMAGES_DIR, image_name) for image_name in image_names]
            for path in paths:
                if os.path.exists(path):
                    if not os.path.exists(os.path.join(category_dir, os.path.basename(path))):
                        print(f"Copying {path} to {category_dir}")
                        shutil.copyfile(path, os.path.join(category_dir, os.path.basename(path)))