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

Build the base conda environment. This will hold the base python installation and core python packages. We want to be able to build this enviornment agnostic of the hardware, operating system, and if our package exists yet. This is why we are not including torch packages or our package in this enviornment yet.

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate cse144-final
```

Now install the specific ML libraries. These will likley be different for each developer, so check what versions work with your system. Here is a sample command for installing torch and torchvision:

```bash
pip install torch torchvision timm
```

Now install our project into the environment. This will allow us to import our package and use it in our code. We install in editable mode so that we can make changes to the code and have them reflected without needing to reinstall the package.

```bash
pip install -e .
```