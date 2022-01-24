#!/bin/sh
sudo docker rm sn-3
system_A_ip = <system_A_ip>
system_C_ip = <system_C_ip>
bash ./swarm-learning/bin/run-sn  \
    --name=sn-1              \
    --host-ip=system_C_ip       \
    --sentinel-ip=system_A_ip   \
    --sn-p2p-port=10000      \
    --sn-api-port=11000      \
    --sn-fs-port=12000       \
    --apls-ip system_A_ip       \
    -serverAddress system_A_ip  \
    -genJoinToken

