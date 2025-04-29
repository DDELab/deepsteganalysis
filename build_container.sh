#!/bin/bash

USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME=$(id -un)

sudo docker build \
  --build-arg USER_ID=$USER_ID \
  --build-arg GROUP_ID=$GROUP_ID \
  --build-arg USERNAME=$USERNAME \
  -t stego .
