#!/bin/sh
sudo docker rm sn-3
#insert variables here:
ip_a=137.226.23.146
ip_c=$(hostname  -I | cut -f1 -d' ')
bash ./swarm-learning/bin/run-sn  \
    --name=sn-3              \
    --host-ip=$ip_c       \
    --sentinel-ip=$ip_a   \
    --sn-p2p-port=16000      \
    --sn-api-port=17000      \
    --sn-fs-port=18000       \
    --sentinel-fs-port=12000 \
    --apls-ip $ip_a       \
    -serverAddress $ip_a  \
    -genJoinToken
