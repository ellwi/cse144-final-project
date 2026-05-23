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


class BaseTransferModel:
    """
    This class serves as a base interface for creating transfer learning models. It defines the structure and expected methods for any transfer learning model we want to implement.
    Subclasses should implement all of the abstract methods defined here to ensure consistency and compatibility with the training pipeline.
    """

    # model attribute should be defined in the subclass's __init__ method, and should be an instance of torch.nn.Module representing the entire model architecture, including the backbone and the classifier head.
    _model: nn.Module

    def __init__(self, num_classes=100):
        """
        MUST IMPLEMENT
        """
        raise NotImplementedError("Subclasses should implement the __init__ method to initialize the model architecture.")

    def get_model(self) -> nn.Module:
        """
        Returns the underlying Pytorch model instance.
        """
        return self._model

    def freeze_all_params(self) -> None:
        """
        Freeze all parameters in the model. This is a default assumption we make because most of our stategy will rely on transfer learning and unfreezing only a few layers.
        """
        for param in self._model.parameters():
            param.requires_grad = False

    def get_classifier_module(self) -> nn.Module:
        """
        MUST IMPLEMENT

        This method should return the specific module within the model that serves as the classification head. 
        This is important for applying different learning rates or unfreezing strategies to the classifier head during training.
        """
        raise NotImplementedError("Subclasses should implement this method to return the classifier module.")
    
    def get_trainable_blocks(self) -> list[nn.Module]:
        """
        MUST IMPLEMENT

        This method should return a list of modules (blocks) in the model that can be unfrozen during training.
        The order of the list should reflect the order in which the blocks should be unfrozen. Index 0 should be the first layer of the backbone, and the last index should be right before the classifier head.
        This allows for a gradual unfreezing strategy during fine-tuning.
        """
        raise NotImplementedError("Subclasses should implement this method to return a list of trainable blocks in order of unfreezing.")
