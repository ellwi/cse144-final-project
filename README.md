University of California Santa Cruz  
CSE 144 Applied Machine Learning Final Project  
Piet Rottinghuis  
Elina Wilson  

### About model  

### Setting up a CSE144 conda environment  
this section should contain package versions and environment setup steps.

### How to use: train.py  
this section should contain the exact commands needed to train the model.

### How to use: inference.py  
this section should contain the exact commands needed to generate `submission.csv`.

## Training and Evaulating Model

Prerequisits:
- Must have docker installed on your machine
- Must have kaggle dataset downloaded and available in the `data/` directory

### Environment setup

Build the docker image:
```bash
docker build -t cse144-final-project:latest .
```

This step may take a few minutes.

The docker image will contain the repository here. When we run it, the base data will exist in /app/data. We will need to
mount our output direcotry over the copied version of `./outputs` in the container so that we can access the generated files
outside of our container.

### Running the training

Run the training using the docker container. It is important to make sure that the shim size is large enough 
for the dataloaders. We set it to 8GB here. We also mount the output directory so that we can access the training output.

```bash
docker run --rm -it --shm-size=8g -v ./cse144-final-project/outputs/:/app/outputs cse144-final:latest \
python train.py --config /app/configs/<config_name>
```

This will generate a new entry into the output/manifest.tsv and create the associated directory with the run. To see the status
of the training in the container see the output/<run_id>/train.log file.


## Evaluate on the training data to generate a sumbission

Now that the training is complete, we will use the container to evaulate the model on the training data.

TODO complete these steps here.


## Developer environment setup

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