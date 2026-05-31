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

Think: “model → predictions
”"""
import torch

def predict(device, net, test_loader, criterion):
    net.eval()
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    test_dataset = test_loader.dataset
    filenames = [test_dataset.__getfilename__(i) for i in range(len(test_dataset))]
    return filenames, all_predictions