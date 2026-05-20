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
pip install torch torchvision
```

Install project package

Finally, install the project in editable mode. This allows the source code in src/ to be imported as a Python package and ensures that changes to the code are immediately reflected without needing to reinstall.
```bash
pip install -e .
```