import os
import wandb
# Constants and directories
DATA_DIR = 'data'
BATCH_SIZE = 32 #no of images processed in each training batch
IMG_HEIGHT, IMG_WIDTH = 64, 64
LATENT_DIM = 100
EPOCHS = 150
CHECKPOINT_DIR = './checkpoints' #folder to save checkpoints
OUTPUT_DIR = './output_images' #folder to save generated images
GIF_PATH = 'training_progress.gif' #file to save the images as gif
RESULTS_DIR = './hyperparameter_results' #folder to store hyperparameter results

# Creating necessary directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initializing W&B
wandb.init(project="gan-dessert-project")