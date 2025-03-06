import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import DEVICE, MODEL_NAME, EVAL_MODEL_PATH, NUM_SAMPLES_TO_SHOW, EVAL_RESULTS_PATH, OUTPUT_HEIGHT, OUTPUT_WIDTH
from dataset import ImageDataset
from torch import nn
from torchvision.utils import save_image
from utils import denormalize, calculate_metrics

# Dynamically import the selected model
MODEL_MODULE = __import__(f"models.{MODEL_NAME}", fromlist=["ScalarToImageModel"])
model = MODEL_MODULE.ScalarToImageModel(output_height=OUTPUT_HEIGHT, output_width=OUTPUT_WIDTH).to(DEVICE)

# Load trained model
if os.path.exists(EVAL_MODEL_PATH):
    model.load_state_dict(torch.load(EVAL_MODEL_PATH, map_location=DEVICE))
    print(f"Loaded model from {EVAL_MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model weights not found at {EVAL_MODEL_PATH}")

model.eval()

# Load dataset (Use validation set for evaluation)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define loss functions
criterion_l1 = nn.L1Loss()

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        from torchvision.models import vgg19
        vgg = vgg19(pretrained=True).features[:8].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.l1_loss(pred_features, target_features)

criterion_perceptual = PerceptualLoss().to(DEVICE)

# Create output directory
os.makedirs(EVAL_RESULTS_PATH, exist_ok=True)

def evaluate_model(model, val_loader):
    total_l1_loss, total_perceptual_loss, total_samples = 0.0, 0.0, 0
    sample_inputs, sample_targets, sample_outputs = [], [], []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Generate predictions
            outputs = model(inputs)

            # Compute loss
            l1_loss = criterion_l1(outputs, targets)
            perceptual_loss = criterion_perceptual(outputs, targets)

            batch_size = inputs.size(0)
            total_l1_loss += l1_loss.item() * batch_size
            total_perceptual_loss += perceptual_loss.item() * batch_size
            total_samples += batch_size

            # Save some samples for visualization
            if len(sample_inputs) < NUM_SAMPLES_TO_SHOW:
                sample_inputs.append(inputs.cpu())
                sample_targets.append(targets.cpu())
                sample_outputs.append(outputs.cpu())

    # Compute average losses
    avg_l1_loss = total_l1_loss / total_samples
    avg_perceptual_loss = total_perceptual_loss / total_samples

    print(f"Evaluation Results: L1 Loss: {avg_l1_loss:.6f}, Perceptual Loss: {avg_perceptual_loss:.6f}")

    # Save images and metrics
    visualize_results(sample_inputs, sample_targets, sample_outputs)
    avg_mae, avg_psnr = calculate_metrics(sample_targets, sample_outputs)

    # Save evaluation results
    with open(os.path.join(EVAL_RESULTS_PATH, "evaluation_results.txt"), "w") as f:
        f.write(f"Model Evaluation Results\n")
        f.write(f"======================\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"- L1 Loss: {avg_l1_loss:.6f}\n")
        f.write(f"- Perceptual Loss: {avg_perceptual_loss:.6f}\n")
        f.write(f"- MAE: {avg_mae:.6f}\n")
        f.write(f"- PSNR: {avg_psnr:.2f} dB\n\n")

    print(f"Evaluation results saved to {EVAL_RESULTS_PATH}")

def visualize_results(inputs, targets, outputs):
    num_samples = len(inputs)
    plt.figure(figsize=(15, 4 * num_samples))

    for i in range(num_samples):
        target_img = targets[i].permute(1, 2, 0).numpy()
        output_img = outputs[i].permute(1, 2, 0).numpy()

        # Plot
        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(target_img)
        plt.title("Target")
        plt.axis('off')

        plt.subplot(num_samples, 2, i*2 + 2)
        plt.imshow(output_img)
        plt.title("Predicted")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_RESULTS_PATH, "evaluation_results.png"), dpi=150)
    plt.show()

# Run evaluation
evaluate_model(model, val_loader)
