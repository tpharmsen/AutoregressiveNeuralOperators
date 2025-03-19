# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script into the container
COPY train.py /app/train.py

# Set the entrypoint to use torchrun for distributed training with correct formatting
ENTRYPOINT ["torchrun", "--nproc_per_node=1", "train.py", "--epochs", "100", "--batch-size", "16"]


#FROM pytorch/pytorch:latest

# Set working directory
#WORKDIR /app

# Copy requirements and install dependencies
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and configuration files
#COPY src/ src/
#COPY conf/ conf/

# Set the command to run training
#CMD ["python", "src/train.py", "--conf", "conf/hpc_01_128_1.yaml"]
