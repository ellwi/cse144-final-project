"""
Job: Define how images are modified before entering the model

This is your augmentation policy module.

Responsibilities:
    Define training transforms:
        resize / crop
        flip
        color jitter
        normalization
    Define validation transforms:
        deterministic resize + normalization
    Define inference transforms (same as validation)
Why this is separate:

So you can easily compare:

no augmentation
weak augmentation
strong augmentation
Mixup later (Milestone 3)

👉 Think: “image preprocessing recipes”
"""