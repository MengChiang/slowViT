# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Use an official PyTorch CUDA-enabled image as a parent image
FROM pytorch/pytorch:1.10.0-cuda11.1-cudnn8-runtime

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Update the system and install git
RUN apt-get update && apt-get install -y git

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Put conda on PATH
ENV PATH=/miniconda/bin:${PATH}

# Create a new conda environment and install PyTorch, transformers
RUN conda create -y -n slowViT python=3.10 && \
    echo "source activate slowViT" > ~/.bashrc && \
    /bin/bash -c "source activate slowViT && conda install -c pytorch pytorch torchvision torchaudio && conda install -c huggingface transformers"

# Clone the SlowFast repository and install it
RUN git clone https://github.com/facebookresearch/SlowFast.git && \
    cd SlowFast && \
    /bin/bash -c "source activate slowViT && pip install -r requirements.txt && python setup.py build develop"

# Change the working directory back to /app
WORKDIR /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run main.py when the container launches
CMD ["python", "src/main.py"]