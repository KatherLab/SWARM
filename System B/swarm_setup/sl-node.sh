#!/bin/sh
sudo docker rm sl-2
#insert variables here:
system_A_ip=<system_A_ip>
data_directory =data
model_directory =MODEL
######
system_B_ip =$(hostname  -I | cut -f1 -d' ')
bash ./swarm-learning/bin/run-sl        \
    --name=sl-1                         \
    --sl-platform=pyt-cv2                   \
    --host-ip=$system_B_ip                 \
    --sn-ip=$system_A_ip                    \
    --sn-api-port=11000                 \
    --sl-fs-port=16000                  \
    --data-dir=$data_directory  \
    --model-dir=$model_directory    \
    --model-program=main.py        \
    --gpu=0                             \
    --apls-ip $system_A_ip                 \
    -serverAddress $system_A_ip            \
    -genJoinToken

