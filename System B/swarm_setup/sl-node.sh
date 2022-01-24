#!/bin/sh
sudo docker rm sl-2
readonly system_B_ip=<system_A_ip>
readonly data_directory = <path-to-data-dir>
readonly model_directory = <path-to-model-dir-having-main-pathon-file/MODEL>
bash ./swarm-learning/bin/run-sl        \
    --name=sl-1                         \
    --sl-platform=pyt-cv2                   \
    --host-ip=system_B_ip                 \
    --sn-ip=system_B_ip                    \
    --sn-api-port=11000                 \
    --sl-fs-port=16000                  \
    --data-dir=data_directory  \
    --model-dir=model_directory    \
    --model-program=main.py        \
    --gpu=0                             \
    --apls-ip system_A_ip                 \
    -serverAddress system_A_ip            \
    -genJoinToken

