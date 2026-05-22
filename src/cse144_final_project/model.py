"""
Builds and returns model from pretrained backbone with all blocks except head frozen.
Also contains unfreeze(block) which does exactly that, used to gradually unfreeze blocks
in train.py to fine-tune model.



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
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch.nn as nn

def unfreeze(blocks):
    """
    Unfreeze all parameters in the specified blocks.
    Usage: 
        model.unfreeze(model.features) to unfreeze all blocks
        model.unfreeze(model.features[-1]) to unfreeze final block before classifier
        ...etc
    """
    for param in blocks.parameters():
        param.requires_grad = True

def build_model(num_classes=100):
    # load pretrained backbone
    print('Building model...')
    print(f'Loading {efficientnet_v2_s}...')
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)

    # replace linear classification head
    # efficientnet classifier: [nn.Sequential, nn.Linear]
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes) # type: ignore # same number of inputs, but new number of classes

    # freeze all features (not-classifier layers)
    for param in model.features.parameters():
        param.requires_grad = False
    
    return model

def main():
    model = build_model()

    print(model)
    print(model.classifier) # view structure

if __name__ == '__main__':
    main()
    