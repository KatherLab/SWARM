#!/bin/sh
sudo docker rm sl-2
ip_1=137.226.23.146
ip_2=$(hostname -I | cut -f1 -d' ') 
model_directory=MODEL
bash ./swarm-learning/bin/run-sl        \
    --name=sl-2                         \
    --sl-platform=pyt-cv2                   \
    --host-ip=$ip_2                 \
    --sn-ip=$ip_2                    \
    --sn-api-port=14000                 \
    --sl-fs-port=20000                  \
    --model-dir=$model_directory    \
    --model-program=main_B.py        \
    --gpu=0                             \
    --apls-ip $ip_1                 \
    -serverAddress $ip_1            \
    -genJoinToken

