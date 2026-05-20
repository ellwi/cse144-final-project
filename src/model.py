"""
Job: Build and configure neural networks

This is your model factory.

Responsibilities:
    Load pretrained backbone (EfficientNet, ConvNeXt, etc.)
    Replace classification head (100 classes)
    Freeze/unfreeze logic (optional early on)
    Return ready-to-train model
Example responsibility:
    build_model(config) → torch.nn.Module
What it SHOULD NOT do:
    Training logic
    Loss computation
    Dataset handling

Think: “model architecture definition only”
"""

class MNISTCNN(nn.Module):
    """Input: (B,1,28,28) -> Output: (B,10)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = None  # Define your convolutional layers here
        self.classifier = None  # Define your fully connected layers here

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )    

    def forward(self, x):
        # Pass input through features, then classifier
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MNISTCNN().to(device)
print(model)


class 