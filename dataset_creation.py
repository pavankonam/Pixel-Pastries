import os
import shutil
import random
# Define paths
FOOD_101_DIR = "/Users/pavankonam/Downloads/food-101"  # Path to the original Food-101 dataset
OUTPUT_DIR = "/Users/pavankonam/Desktop/Gen AI"  # Path to save the filtered dataset

# Revised list of dessert classes
DESSERT_CLASSES = [
    "apple_pie", "bread_pudding", "cannoli", "carrot_cake", "cheesecake",
    "chocolate_cake", "chocolate_mousse", "cup_cakes", "donuts", "ice_cream",
    "macarons", "pancakes", "red_velvet_cake", "strawberry_shortcake", "waffles", "tiramisu"
]

# Create output directories
os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

# Read train/test splits
def read_split(split_file):
    with open(split_file, "r") as f:
        return [line.strip() for line in f]

train_files = read_split(os.path.join(FOOD_101_DIR, "meta", "train.txt"))
test_files = read_split(os.path.join(FOOD_101_DIR, "meta", "test.txt"))

# Function to copy files for a specific split
def copy_files(split_files, split_name):
    for file in split_files:
        class_name = file.split("/")[0]
        if class_name in DESSERT_CLASSES:
            src = os.path.join(FOOD_101_DIR, "images", f"{file}.jpg")
            dst_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f"{file.split('/')[-1]}.jpg")
            shutil.copy(src, dst)

# Function to split train into train and validation
def split_train_val(train_files, val_ratio=0.2):
    train_split, val_split = {}, {}
    for file in train_files:
        class_name = file.split("/")[0]
        if class_name not in train_split:
            train_split[class_name] = []
        train_split[class_name].append(file)
    
    for class_name, files in train_split.items():
        random.shuffle(files)  # Shuffle the files for randomness
        split_idx = int(len(files) * (1 - val_ratio))
        train_split[class_name] = files[:split_idx]
        val_split[class_name] = files[split_idx:]
    
    return train_split, val_split

# Split train into train and validation
train_split, val_split = split_train_val(train_files)

# Flatten the splits back into lists
train_files = [file for files in train_split.values() for file in files]
val_files = [file for files in val_split.values() for file in files]

# Copy train, val, and test files
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("Data extraction complete!")