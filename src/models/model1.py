## ****** Model 1 ******
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19

class ScalarToImageModel(nn.Module):
    def __init__(self, output_height=128, output_width=256):
        super(ScalarToImageModel, self).__init__()

        initial_h, initial_w = output_height // 64, output_width // 64

        self.fc = nn.Sequential(
            nn.Linear(3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * initial_h * initial_w),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.initial_h = initial_h
        self.initial_w = initial_w

        self.deconv_module = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, self.initial_h, self.initial_w)
        return self.deconv_module(x)
