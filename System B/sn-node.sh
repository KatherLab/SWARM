#!/bin/sh
sudo docker rm sn-2
ip_a=137.226.23.146
ip_b=$(hostname -I | cut -f1 -d' ') 
bash ./swarm-learning/bin/run-sn      --name=sn-2                  --host-ip=$ip_b          --sentinel-ip=$ip_a      --sn-p2p-port=13000          --sn-api-port=14000          --sn-fs-port=15000           --sentinel-fs-port=12000     --apls-ip $ip_a          -serverAddress $ip_a     -genJoinToken
