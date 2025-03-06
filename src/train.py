import torch
import os
from torch.utils.data import DataLoader
from config import DEVICE, BATCH_SIZE, MODEL_NAME, MODEL_SAVE_PATH, EPOCHS, PATIENCE
from dataset import ImageDataset
from torch import optim
import torch.nn as nn

# Dynamically import the selected model
MODEL_MODULE = __import__(f"models.{MODEL_NAME}", fromlist=["ScalarToImageModel"])
model = MODEL_MODULE.ScalarToImageModel().to(DEVICE)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Load dataset
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = sum(criterion(model(x.to(DEVICE)), y.to(DEVICE)).item() for x, y in train_loader) / len(train_loader)
    
    model.eval()
    val_loss = sum(criterion(model(x.to(DEVICE)), y.to(DEVICE)).item() for x, y in val_loader) / len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break
