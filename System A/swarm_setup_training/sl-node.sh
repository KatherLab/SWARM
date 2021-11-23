#!/bin/sh
sudo docker rm sl-1
bash ./swarm-learning/bin/run-sl        \
    --name=sl-1                         \
    --sl-platform=pyt-cv2                   \
    --host-ip= system_A_ip                  \
    --sn-ip= system_A_ip                  \
    --sn-api-port=11000                 \
    --sl-fs-port=16000                  \
    --data-dir=path-to-data-dir \
    --model-dir=path-to-model-dir-having-main-pathon-file/model    \
    --model-program=main.py        \
    --gpu=0                             \
    --apls-ip  system_A_ip               \
    -serverAddress system_A_ip            \
    -genJoinToken

