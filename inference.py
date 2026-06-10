"""
Script entry point for running inference on the test dataset.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import logging.config

from cse144_final_project.dataset import get_test_dataloader
from cse144_final_project.model import build_model
from cse144_final_project.inference import predict
from cse144_final_project.utils import set_seed


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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--outfile",
        type=Path,
        default="./outputs/submission.csv",
        help="Path to output submission CSV"
    )

    parser.add_argument(
        "--model", 
        type=str, 
        default='EfficientNet_V2_S', 
        help="Pretrained model to use for transfer learning. Options are defined in model.py."
    )

    return parser.parse_args()


def main():
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(levelname)s: %(message)s"
            },
            "detailed": {
                "format": "[%(levelname)s|%(module)s|%(lineno)d] %(asctime)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z"  # ISO 8601 with timezone
            }
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["stdout"]
        }
    }
    logging.config.dictConfig(config=logging_config)

    logger = logging.getLogger("inference")

    args = parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    # Ensure output directory exists
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    # Build test dataloader
    test_loader = get_test_dataloader(
        data_dir=args.testdir,
        model =args.model,
        batch_size=32,
        num_workers=2,
        logger=logger
    )

    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Build model
    model = build_model(args.model, num_classes=100, logger=logger)

    # Load trained weights
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    # Run inference
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    filenames, predictions = predict(device, model, test_loader, criterion)

    # Write submission CSV
    with open(args.outfile, "w") as f:
        f.write("ID,Label\n")

        for image_id, pred in zip(filenames, predictions):
            f.write(f"{image_id},{pred}\n")

    logger.info("\nInference complete.")
    logger.info(f"Submission saved to: {args.outfile}")


if __name__ == "__main__":
    main()