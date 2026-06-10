FROM python:3.10-slim

WORKDIR /app

# upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# install PyTorch
RUN pip install --no-cache-dir \
    torch==2.12.0+cu126 \
    torchvision==0.27.0+cu126 \
    torchaudio==2.11.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# install dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.3.3 \
    matplotlib==3.10.9 \
    scikit-learn==1.7.2
COPY . .

# Download the package into pip
RUN pip install -e .