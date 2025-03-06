import torch
import os

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "../data/Cleaned_CNN_data.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "../data/resized_images")

# Model Selection
MODEL_NAME = "model1"  # Change to model2, model3, etc.
MODEL_SAVE_PATH = os.path.join(BASE_DIR, f"../experiments/{MODEL_NAME}/best_model.pth")
EVAL_MODEL_PATH = MODEL_SAVE_PATH  # Path for evaluation

# Normalization Constants
TEMP_MIN, TEMP_MAX = 290, 299
VEL_MIN, VEL_MAX = 1.75, 2.65
POWER_MIN, POWER_MAX = 417, 1667

# Training Parameters
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
EPOCHS = 50
PATIENCE = 5

# Evaluation Parameters
NUM_SAMPLES_TO_SHOW = 5
OUTPUT_HEIGHT, OUTPUT_WIDTH = 128, 256
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, f"../experiments/{MODEL_NAME}/evaluations/")
