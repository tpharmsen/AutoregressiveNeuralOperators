# Use the official PyTorch image with CUDA support
#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM pytorch/pytorch:latest
# nvcr.io/nvidia/pytorch:25.02-py3-igpu

# Set the working directory
WORKDIR /home/

# Install required Python packages
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Copy your script into the container
#COPY . /home/

# Set the entrypoint to use torchrun for distributed training with correct formatting
#ENTRYPOINT ["torchrun", "--nproc_per_node=1", "src/train.py", "--epochs", "100", "--batch-size", "16"]
CMD ["python", "AutoregressiveNeuralOperators/src/train.py", "--conf", "conf/local_128_1.yaml"]

