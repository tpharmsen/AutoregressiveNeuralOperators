# Use the official PyTorch image with CUDA support
#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM pytorch/pytorch:latest

# Set the working directory
#WORKDIR /app

# Install required Python packages
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Copy your script into the container
#COPY /src/train.py /train.py

# Set the entrypoint to use torchrun for distributed training with correct formatting
#ENTRYPOINT ["torchrun", "--nproc_per_node=1", "src/train.py", "--epochs", "100", "--batch-size", "16"]
CMD ["python", "src/train.py", "--conf", "conf/hpc_01_128_1.yaml"]


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
