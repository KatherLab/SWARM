# SWARM
# SWARM Learning For Histopathology Image Analysis

This repository contains requirements for running swarm learing on for workflow for end-to-end artificial intelligence on histopathology images. It is based on workflows which were previously described in [Warnat-Herresthal et al., Nature 2021](https://rdcu.be/cA9XP) and [Laleh, N. G. et al., Gastroenterology 2020](https://www.biorxiv.org/content/10.1101/2021.08.09.455633v1.full.pdf). The objective is to run a decentralized training on multiple cohorts in multiple centers and predict a given *label* directly from digitized histological whole slide images (WSI). The *label* is defined on the level of *patients*, not on the level of pixels in a given WSI. Thus, the problems addressed by HAI are *weakly supervised problems*.

This is important to notice that there are various changes in this version but it follows the same steps.

++ These scripts are still under the development and please always use the final version of it ++

## How to use this repository:
We assume that there are three physically separate systems (System A, System B, System C) used for the SL framework in our approach.
We assume that there are three physically separate systems (System A, System B, System C) used for the SL framework in our approach. All images were available in ScanScope Virtual Slide (SVS) format and were tessellated using https://github.com/KatherLab/preProcessing according to the “The Aachen Protocol for Deep Learning Histopathology: A hands-on guide for data preprocessing”.
 System A is used to initialize the licence server by starting the licence container and installing the swarm licence downloaded from the HPE login website. System A also starts the SPIFFE SPIRE container. The first SN process (node) to go online is referred to as the “sentinel” node and will be the first to register itself with the SPIFFE SPIRE network. When the SN node of the system A is ready the, SN node of System B and System C are run. During training each system trains its data batch in the local system till the merging criterion (sync interval) is reached. The node which finishes its training batch first will be the leader and will collect the learning from other peers (depending on the minimum number of peers, in our case two), average the learning weights and send it back. Since the data samples are of different sizes the SL node stops at different instances creating different checkpoint models
To use this workflow, you need to modfiy specific files based on your project.

System A, B, C has two subfolders having *single_model_training* containing  the programs needed training the histopathology Image analysis based on [Laleh, N. G. et al., Gastroenterology 2020](https://www.biorxiv.org/content/10.1101/2021.08.09.455633v1.full.pdf) and *swarm_setup_training* containing *MODEL* folder having the code for running the swarm learning node and also 
Experiment file is a text file and an example of it can be find this repository.
## Run training :

To start training, we use the Main.py script. The full path to the experiemnt file, should be used as an input variable in this script.
