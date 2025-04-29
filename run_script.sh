sudo docker run \
    --gpus all -it --shm-size=16G \
    -v "/mnt/hdd:/mnt/hdd" \
    -v "$PWD:$HOME/deepsteganalysis" \
    -v "$HOME/LogFiles:$HOME/LogFiles" \
    -e WANDB_API_KEY=$WANDB_API_KEY stego \
    bash -c "umask 0002 && python3 ./train_lit_model.py"
