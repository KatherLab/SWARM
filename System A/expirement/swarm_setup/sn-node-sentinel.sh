#!/bin/sh
sudo docker rm sn-1
bash ./swarm-learning/bin/run-sn  \
    --name=sn-1              \
    --host-ip=137.226.23.146      \
    --sentinel-ip=137.226.23.146  \
    --sn-p2p-port=10000      \
    --sn-api-port=11000      \
    --sn-fs-port=12000       \
    --apls-ip 137.226.23.146      \
    -serverAddress 137.226.23.146 \
    -genJoinToken

