#!/bin/sh
sudo docker rm sl-1
bash ./swarm-learning/bin/run-sl        \
    --name=sl-1                         \
    --sl-platform=pyt-cv2                   \
    --host-ip=137.226.23.146                  \
    --sn-ip=137.226.23.146                    \
    --sn-api-port=11000                 \
    --sl-fs-port=16000                  \
    --data-dir=examples/test-example/app-data  \
    --model-dir=examples/test-example/model    \
    --model-program=TCGA.py        \
    --gpu=0                             \
    --apls-ip 137.226.23.146                 \
    -serverAddress 137.226.23.146            \
    -genJoinToken

