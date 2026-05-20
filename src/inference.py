"""
Job: Turn a trained model into predictions

This is your evaluation engine for test data.

Responsibilities:
    Load trained checkpoint
    Set model to eval mode
    Run forward pass on test images
    Collect raw outputs (logits or probabilities)
What it should NOT do:
    No CSV formatting (that’s submission.py or predict.py)
    No training logic

👉 Think: “model → predictions
”"""