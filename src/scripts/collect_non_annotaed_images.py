import os
import shutil

IMAGES_DIR = "/mnt/e/Collected_non_labled/Sample_2"
OUTPUT_DIR = "/mnt/e/Collected_non_labled/Sample_2_aggregated_images"

image_paths = set([os.path.join(IMAGES_DIR, img) for img in os.listdir(IMAGES_DIR) if img.endswith((".jpg", ".jpeg", ".png"))])
pbi_paths = set([os.path.join(OUTPUT_DIR, "PBI", img) for img in os.listdir(os.path.join(OUTPUT_DIR, "PBI")) if img.endswith((".jpg", ".jpeg", ".png"))])
streamer_paths = set([os.path.join(OUTPUT_DIR, "Streamer", img) for img in os.listdir(os.path.join(OUTPUT_DIR, "Streamer")) if img.endswith((".jpg", ".jpeg", ".png"))])

image_basenames = set([os.path.basename(path) for path in image_paths])

pbi_basenames = set([os.path.basename(path) for path in pbi_paths])
streamer_basenames = set([os.path.basename(path) for path in streamer_paths])

no_pbi_basenames = image_basenames - pbi_basenames
no_streamer_basenames = image_basenames - streamer_basenames

no_pbi_paths = [os.path.join(IMAGES_DIR, img) for img in no_pbi_basenames]
no_streamer_paths = [os.path.join(IMAGES_DIR, img) for img in no_streamer_basenames]

print(f"Found {len(no_pbi_paths)} images without PBI annotations.")
print(f"Found {len(no_streamer_paths)} images without Streamer annotations.")

for path in no_pbi_paths:
    print(f"Processing {path} for No_PBI category.")
    os.makedirs(os.path.join(OUTPUT_DIR, "No_PBI"), exist_ok=True)
    if os.path.exists(path):
        if not os.path.exists(os.path.join(OUTPUT_DIR, "No_PBI", os.path.basename(path))):
            print(f"Copying {path} to {os.path.join(OUTPUT_DIR, 'No_PBI')}")
            shutil.copyfile(path, os.path.join(OUTPUT_DIR, "No_PBI", os.path.basename(path)))

for path in no_streamer_paths:
    os.makedirs(os.path.join(OUTPUT_DIR, "No_Streamer"), exist_ok=True)
    if os.path.exists(path):
        if not os.path.exists(os.path.join(OUTPUT_DIR, "No_Streamer", os.path.basename(path))):
            print(f"Copying {path} to {os.path.join(OUTPUT_DIR, 'No_Streamer')}")
            shutil.copyfile(path, os.path.join(OUTPUT_DIR, "No_Streamer", os.path.basename(path)))