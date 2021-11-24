#!/bin/sh
sudo docker rm sn-2
bash ./swarm-learning/bin/run-sn  \
    --name=sn-1              \
    --host-ip=system_B_ip       \
    --sentinel-ip=system_A_ip   \
    --sn-p2p-port=10000      \
    --sn-api-port=11000      \
    --sn-fs-port=12000       \
    --apls-ip system_A_ip       \
    -serverAddress system_A_ip  \
    -genJoinToken
