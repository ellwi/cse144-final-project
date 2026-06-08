University of California Santa Cruz  
CSE 144 Applied Machine Learning Final Project  
Piet Rottinghuis  
Elina Wilson  

## Final submission instructions

Prerequisites:
- Must have [Docker](https://www.docker.com/) installed on your machine
- Must have the Kaggle dataset downloaded and available in the `data/` directory in the project root.

*Note: The docker image is designed to run on a CUDA system.*

### Environment setup
Navigate to the project root directory and build the docker image:
```bash
cd ~/.../cse144-final-project
docker build -t cse144-final-project:latest .
```

This step may take a multiple minutes. The installation of CUDA and PyTorch is the most time consuming step.

The docker image will contain this repository. When we run it, the training and testing data will exist in /app/data. We will need to
mount our output direcotry over the copied version of `./outputs`.

### Running the training

Run the training using the docker container. It is important to make sure that the shared memory size is large enough 
for the dataloaders. We set it to 8GB here. We also mount the output directory so that we can access the training output.

TODO generate a final submission config and update the command below to use it.

```bash
docker run --rm --shm-size=8g -v ./outputs/:/app/outputs cse144-final:latest \
python train.py --config /app/configs/<config_name>
```

This will generate a new entry into the output/manifest.tsv and create the associated directory with the run. To see the status
of the training in the container see the output/<run_id>/train.log file.

Example manifest entry after 2 training runs:
```
run_id	timestamp	model_name	epochs	batch_size	run_dir
run_20260604_213526_fe56	2026-06-04T21:35:26.593249	EfficientNet_V2_S	20	15	outputs/run_20260604_213526_fe56
run_20260608_000035_2ca3	2026-06-08T00:00:35.335291	EfficientNet_V2_S	60	10	outputs/run_20260608_000035_2ca3
```

Example output directory structure after 2 training runs:
```
outputs/
├── manifest.tsv
├── run_20260604_213526_fe56/
│   ├── best_checkpoint.pth
│	├── config.yaml
│	├── history.tsv
│	├── last_checkpoint.pth
│	├── metrics.json
│	├── training.log
│	└── training_plots.png
└── run_20260608_000035_2ca3/
	├── best_checkpoint.pth
	├── config.yaml
	├── history.tsv
	├── last_checkpoint.pth
	├── metrics.json
	├── training.log
	└── training_plots.png
```

## Evaluate on the training data to generate a sumbission

Now that the training is complete, we will use the container to evaulate the model on the training data.

Look in output/manifest.tsv to find the run_id of the training run you want to evaluate. Then run the following command, replacing <run_id> with the actual run_id.
Optionally adjust the `--outfile` argument to specify the name of the output file. Remember that the output file path will be relative to the container and will only
be accessible if written to a mounted directory (like `./outputs`).

```bash
docker run --rm --shm-size=8g -v ./outputs/:/app/outputs cse144-final:latest \
python inference.py --checkpoint /app/outputs/<run_id>/best_checkpoint.pth --outfile /app/outputs/final_submission.csv
```

Find the generated `final_submission.csv` file in the path provided to the `--outfile` argument. This file can be submitted to Kaggle for evaluation.

## Developer environment setup

Instructions for setting up a local development environment. This is useful for debugging and iterating on the code outside of the docker container.

We first create a base conda environment. This environment contains the Python installation and core scientific packages. It is designed to be independent of the project code and does not include deep learning frameworks such as PyTorch.

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate cse144-final
```

Install PyTorch (GPU-enabled backend)

PyTorch is installed separately because its installation depends on system-specific CUDA configuration. This ensures compatibility across different machines (e.g., local GPU machines, Colab, or lab servers).
```bash
python -m pip install torch torchvision
```

Install project package

Finally, install the project in editable mode. This allows the source code in src/ to be imported as a Python package and ensures that changes to the code are immediately reflected without needing to reinstall.
```bash
python -m pip install -e .
```