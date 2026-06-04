FROM python:3.10-slim

WORKDIR /app

# upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# install PyTorch
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# install dependencies
RUN pip install --no-cache-dir \
    "numpy<2" \
    pandas \
    matplotlib \
    scikit-learn

COPY . .

# Download the package into pip
RUN pip install -e .