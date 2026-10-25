import torch.nn as nn

# CNN class
class CNNFromScratch(nn.Module):
    def __init__(self, num_classes = 7):
        super(CNNFromScratch, self).__init__()
        self.features = nn.Sequential(
            # Layer 1 3 -> 32 | 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            # Layer 2 32 -> 64 | 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 3 64 -> 128 | 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 4 128 -> 256 | 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Average pooling 14 -> 1
            nn.AdaptiveAvgPool2d((1,1))
            # Output dimensions 256 x 1 x 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLu(),
            nn.Droupout(0.5),
            nn.Linear(128, num_classes)
        )
    # forward method
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x