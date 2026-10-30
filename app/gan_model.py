import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=2, padding=1
        )
        self.tanh = nn.Tanh()

    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(noise.size(0), 128, 7, 7)
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.tanh(self.deconv2(x))
        return x