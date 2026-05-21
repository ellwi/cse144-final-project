"""
Script entry point for running inference on the test dataset.
"""

import argparse
from pathlib import Path

import torch

from cse144_final_project.dataset import get_test_dataloader
from cse144_final_project.model import build_model
from cse144_final_project.inference import predict
from cse144_final_project.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on the CSE144 test dataset"
    )

    parser.add_argument(
        "--testdir",
        type=Path,
        default="./data/test",
        help="Path to test image directory"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--outfile",
        type=Path,
        default="./outputs/submission.csv",
        help="Path to output submission CSV"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Ensure output directory exists
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    # Build test dataloader
    test_loader = get_test_dataloader(
        test_dir=args.testdir,
        batch_size=32,
        num_workers=2
    )

    print(f"Test samples: {len(test_loader.dataset)}")

    # Build model
    model = build_model(num_classes=100, pretrained=False)

    # Load trained weights
    load_checkpoint(model, args.checkpoint)

    model = model.to(device)

    # Run inference
    predictions = predict(
        model=model,
        dataloader=test_loader,
        device=device
    )

    # Retrieve image IDs from dataset
    image_ids = test_loader.dataset.image_ids

    # Write submission CSV
    with open(args.outfile, "w") as f:
        f.write("ID,Label\n")

        for image_id, pred in zip(image_ids, predictions):
            f.write(f"{image_id},{pred}\n")

    print("\nInference complete.")
    print(f"Submission saved to: {args.outfile}")


if __name__ == "__main__":
    main()