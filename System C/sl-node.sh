#!/bin/sh
sudo docker rm sl-3
model_directory=MODEL
ip_a=137.226.23.146
ip_c=$(hostname  -I | cut -f1 -d' ')
bash ./swarm-learning/bin/run-sl        \
    --name=sl-3                         \
    --sl-platform=pyt-cv2                   \
    --host-ip=$ip_c                 \
    --sn-ip=$ip_c                   \
    --sn-api-port=17000                 \
    --sl-fs-port=19000                  \
    --model-dir=$model_directory    \
    --model-program=main_c.py        \
    --gpu=0                             \
    --apls-ip $ip_a                 \
    -serverAddress $ip_a            \
    -genJoinToken

