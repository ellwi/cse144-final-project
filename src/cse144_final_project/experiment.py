from pathlib import Path
from datetime import datetime
import uuid
import json
import csv
import yaml

import pandas as pd


class ExperimentRun:
    """
    Manages experiment output directories, metadata, and artifact saving.

    Expected output structure:

    outputs/
    ├── manifest.tsv
    └── run_20260601_153012_ab12/
        ├── config.yaml
        ├── history.tsv
        ├── metrics.json
        └── ...
    """

    def __init__(self, config):
        self.config = config

        # Root output directory
        self.output_root = config.output.out_dir
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Generate unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:4]

        self.run_id = f"run_{timestamp}_{short_uuid}"

        # Create run directory
        self.run_dir = self.output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

        # Common artifact paths
        self.config_path = self.run_dir / "config.yaml"
        self.history_path = self.run_dir / "history.tsv"
        self.metrics_path = self.run_dir / "metrics.json"

        # Manifest file
        self.manifest_path = self.output_root / "manifest.tsv"

    def save_config(self):
        """
        Save the fully resolved config used for this run.
        """

        with open(self.config_path, "w") as f:
            yaml.safe_dump(
                self.config.model_dump(mode="json"),
                f,
                sort_keys=False
            )

    def save_history(self, history):
        """
        Save epoch-by-epoch training history.

        Expected history format:
        {
            "train_loss": [...],
            "train_acc": [...],
            "val_loss": [...],
            "val_acc": [...]
        }
        """

        df = pd.DataFrame(history)

        # Add epoch column
        df.insert(0, "epoch", range(1, len(df) + 1))

        df.to_csv(self.history_path, index=False, sep='\t')

    def save_metrics(self, history):
        """
        Save summary metrics for the epoch of the experiment with the best validation accuracy.
        """

        best_epoch = max(
            range(len(history["val_acc"])),
            key=lambda i: history["val_acc"][i]
        )

        metrics = {
            "run_id": self.run_id,
            "best_epoch": best_epoch + 1,
            "best_val_acc": history["val_acc"][best_epoch],
            "best_val_loss": history["val_loss"][best_epoch],
            "train_acc_at_best": history["train_acc"][best_epoch],
            "train_loss_at_best": history["train_loss"][best_epoch],
        }

        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    def save_manifest_entry(self):
        """
        Append a summary entry for this run to manifest.tsv.
        """

        manifest_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": self.config.model.model_name,
            "epochs": self.config.training.epochs,
            "batch_size": self.config.data.batch_size,
            "run_dir": str(self.run_dir),
        }

        file_exists = self.manifest_path.exists()

        with open(self.manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=manifest_entry.keys(),
                delimiter='\t'
            )

            # Write header if file does not exist
            if not file_exists:
                writer.writeheader()

            writer.writerow(manifest_entry)
