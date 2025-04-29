FROM python:3.12-slim

ARG USER_ID
ARG GROUP_ID
ARG USERNAME

RUN if [ -z "$USER_ID" ]; then echo "USER_ID is required"; exit 1; fi && \
    if [ -z "$GROUP_ID" ]; then echo "GROUP_ID is required"; exit 1; fi && \
    if [ -z "$USERNAME" ]; then echo "USERNAME is required"; exit 1; fi

# install build utilities
RUN apt-get update
RUN apt-get install -y ca-certificates
RUN apt-get install -y gcc make apt-transport-https build-essential git
RUN apt-get install ffmpeg libsm6 libxext6  -y

# check python environment
RUN python3 --version
RUN pip3 --version

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/dwgoon/jpegio

# Add user and group
RUN groupadd -g "${GROUP_ID}" "${USERNAME}" && \
    useradd -m -u "${USER_ID}" -g "${GROUP_ID}" -s /bin/bash "${USERNAME}"

USER ${USERNAME}
RUN mkdir -p /home/${USERNAME}/deepsteganalysis && chown ${USER_ID}:${GROUP_ID} /home/${USERNAME}/deepsteganalysis
RUN mkdir -p /home/${USERNAME}/LogFiles && chown ${USER_ID}:${GROUP_ID} /home/${USERNAME}/LogFiles
WORKDIR /home/${USERNAME}/deepsteganalysis

# Running Python Application
CMD ["python3", "./train_lit_model.py"]
