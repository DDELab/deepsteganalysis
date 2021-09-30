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
RUN useradd -ms /bin/bash stego
USER stego
WORKDIR  /home/stego

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/dwgoon/jpegio

# Copy all the files from the project’s root to the working directory
COPY . deepsteganalysis
WORKDIR /home/stego/deepsteganalysis

# Running Python Application
CMD ["python3", "./train_lit_model.py", "logging.wandb.activate=False", "training.epochs=1"]
