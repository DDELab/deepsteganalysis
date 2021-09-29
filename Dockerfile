#FROM pytorchlightning/pytorch_lightning:base-conda-py3.6-torch1.9
#FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# install build utilities
RUN apt-get update
RUN apt-get install -y ca-certificates 
RUN apt-get install -y gcc make apt-transport-https build-essential git 
RUN apt-get install ffmpeg libsm6 libxext6  -y


# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR  /usr/src/deepsteganalysis

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/dwgoon/jpegio

# Copy all the files from the project’s root to the working directory
COPY src/ /src/
RUN ls -la /src/*

# Running Python Application
CMD ["python3", "/src/train_lit_model.py"]
