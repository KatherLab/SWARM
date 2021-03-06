#!/bin/sh
sudo docker rm sl-1
#add variables here:
model_directory=MODEL
#####
system_A_ip=$(hostname  -I | cut -f1 -d' ')
bash ./swarm-learning/bin/run-sl        \
    --name=sl-1                         \
    --sl-platform=pyt-cv2                   \
    --host-ip=$system_A_ip                  \
    --sn-ip=$system_A_ip                  \
    --sn-api-port=11000                 \
    --sl-fs-port=16000                  \
    --model-dir=$model_directory    \
    --model-program=main_A.py        \
    --gpu=0                             \
    --apls-ip  $system_A_ip              \
    -serverAddress $system_A_ip           \
    -genJoinToken

