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
from abc import ABC, abstractmethod

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch.nn as nn


def freeze_all_params(model: nn.Module) -> None:
    """
    Freeze all parameters in the model.
    """
    for param in model.parameters():
        param.requires_grad = False


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

    # freeze all layers of the model
    freeze_all_params(model)
    
    return model


class BaseTransferModel(nn.Module, ABC):
    """
    This class serves as a base interface for creating transfer learning models. It defines the structure and expected methods for any transfer learning model we want to implement.
    Subclasses should implement all of the abstract methods defined here to ensure consistency and compatibility with the training pipeline.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Do not call forward directly. Use 'model(x)' instead, which will call this method under the hood. This is a standard PyTorch convention.
        """
        return self.model(x)

    def freeze_all_params(self) -> None:
        """
        Freeze all parameters in the model. This is a default assumption we make because most of our stategy will rely on transfer learning and unfreezing only a few layers.
        """
        for param in self.parameters():
            param.requires_grad = False

    @abstractmethod
    def get_classifier_module(self) -> nn.Module:
        """
        This method should return the specific module within the model that serves as the classification head. 
        This is important for applying different learning rates or unfreezing strategies to the classifier head during training.
        """
        pass

    @abstractmethod
    def get_trainable_backbone_blocks(self) -> list[nn.Module]:
        """
        This method should return a list of modules (blocks) from the model backbone. Don't return classifier head blocks here.
        The order of the list should reflect the order in which the blocks should be unfrozen. Index 0 should be the first layer of the backbone, and the last index should be right before the classifier head.
        This allows for a gradual unfreezing strategy during fine-tuning.
        """
        pass


class EfficientNetV2STM(BaseTransferModel):
    """
    EfficientNetV2-S transfer learning model for CSE144.
    """

    def __init__(self, num_classes=100):
        super().__init__()

        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.model = efficientnet_v2_s(weights=weights)

        # replace linear classification head
        # efficientnet classifier: [nn.Sequential, nn.Linear]
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes) # type: ignore # same number of inputs, but new number of classes

    def get_classifier_module(self) -> nn.Module:
        return self.model.classifier

    def get_trainable_backbone_blocks(self) -> list[nn.Module]:
        return list(self.model.features)