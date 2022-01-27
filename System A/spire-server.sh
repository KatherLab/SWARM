#!/bin/sh
sudo docker rm spire-server
bash ./swarm-learning/bin/run-spire-server --name=spire-server -p 8081:8081
