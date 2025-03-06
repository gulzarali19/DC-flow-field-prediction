import pandas as pd
import os
import numpy as np
from config import DATA_CSV, IMAGE_DIR, TEMP_MIN, TEMP_MAX, VEL_MIN, VEL_MAX, POWER_MIN, POWER_MAX

# Load CSV
df = pd.read_csv(DATA_CSV)

# Check required columns
required_columns = {"Power", "Temp", "Velocity", "Image"}
if not required_columns.issubset(df.columns):
    raise ValueError("CSV must contain 'Power', 'Temp', 'Velocity', 'Image' columns.")

# Image Paths & Scalars
image_paths, temp_data, vel_data, power_data = [], [], [], []

for _, row in df.iterrows():
    img_name = str(row["Image"]).strip()
    img_name = img_name if img_name.lower().endswith(".png") else img_name + ".png"
    img_path = os.path.join(IMAGE_DIR, img_name)

    if os.path.exists(img_path):
        image_paths.append(img_path)
        temp_data.append(row["Temp"])
        vel_data.append(row["Velocity"])
        power_data.append(row["Power"])
    else:
        print(f"Warning: Image {img_path} not found. Skipping.")

# Convert to NumPy arrays & Normalize
def min_max_normalize(data, min_val, max_val):
    return 2 * (data - min_val) / (max_val - min_val) - 1

temperature_data = min_max_normalize(np.array(temp_data).reshape(-1, 1), TEMP_MIN, TEMP_MAX)
velocity_data = min_max_normalize(np.array(vel_data).reshape(-1, 1), VEL_MIN, VEL_MAX)
power_data = min_max_normalize(np.array(power_data).reshape(-1, 1), POWER_MIN, POWER_MAX)

scalar_inputs = np.concatenate([temperature_data, velocity_data, power_data], axis=1)
