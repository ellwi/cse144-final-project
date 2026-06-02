FROM python:3.10-slim

WORKDIR /app

# upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# install PyTorch CPU wheels (IMPORTANT: stable official source)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# install dependencies
RUN pip install --no-cache-dir \
    "numpy<2" \
    pandas \
    matplotlib \
    scikit-learn

COPY . .

# Download the package into pip
RUN pip install -e .