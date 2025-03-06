import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from config import BATCH_SIZE

class ImageDataset(Dataset):
    def __init__(self, inputs, image_paths, target_height=128, target_width=256):
        self.inputs = inputs
        self.image_paths = image_paths
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_width, self.target_height))
        img = torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)
        return input_data, img
