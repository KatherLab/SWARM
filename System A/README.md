# SWARM Learning For Histopathology Image Analysis

This repository contains requirements for running swarm learing on for workflow for end-to-end artificial intelligence on histopathology images. It is based on workflows which were previously described in [Warnat-Herresthal et al., Nature 2021](https://rdcu.be/cA9XP) and [Laleh, N. G. et al., Gastroenterology 2020](https://www.biorxiv.org/content/10.1101/2021.08.09.455633v1.full.pdf). The objective is to run a decentralized training on multiple cohorts in multiple centers and predict a given *label* directly from digitized histological whole slide images (WSI). The *label* is defined on the level of *patients*, not on the level of pixels in a given WSI. Thus, the problems addressed by HAI are *weakly supervised problems*.

This is important to notice that there are various changes in this version but it follows the same steps.

++ These scripts are still under the development and please always use the final version of it ++

## How to use this repository:
The first step is 
To use this workflow, you need to modfiy specific experiement file based on your project. Experiment file is a text file and an example of it can be find this repository. For this file you need to fill the following options:


## Run training :

To start training, we use the Main.py script. The full path to the experiemnt file, should be used as an input variable in this script.
