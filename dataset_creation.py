import os
import shutil
import random

FOOD_101_DIR = "/Users/pavankonam/Downloads/food-101"  # Path to the original Food-101 dataset
OUTPUT_DIR = "/Users/pavankonam/Desktop/Gen AI"  # Your working directory

# Selected desserts
DESSERT_CLASSES = [
    "apple_pie", "bread_pudding", "cannoli", "carrot_cake", "cheesecake",
    "chocolate_cake", "chocolate_mousse", "cup_cakes", "donuts", "ice_cream",
    "macarons", "pancakes", "red_velvet_cake", "strawberry_shortcake", "waffles", "tiramisu"
]

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to move files for a specific class
def move_dessert_classes():
    for class_name in DESSERT_CLASSES:
        src_dir = os.path.join(FOOD_101_DIR, "images", class_name)
        dst_dir = os.path.join(OUTPUT_DIR, class_name)
        
        # Check if the source directory exists
        if os.path.exists(src_dir):
            # Move the entire folder to the output directory
            shutil.move(src_dir, dst_dir)
            print(f"Moved folder: {class_name}")
        else:
            print(f"Folder not found: {class_name}")

# Move the dessert folders
move_dessert_classes()

print("Data extraction complete!")
