# SWARM
# HIA (Histopathology Image Analysis)

This repository contains requirements for running swarm learing on for workflow for end-to-end artificial intelligence on histopathology images. It is based on workflows which were previously described in [Warnat-Herresthal et al., Nature 2021](https://rdcu.be/cA9XP) and [Laleh, N. G. et al., Gastroenterology 2020](https://www.biorxiv.org/content/10.1101/2021.08.09.455633v1.full.pdf). The objective is to run a decentralized training on multiple cohorts in multiple centers and predict a given *label* directly from digitized histological whole slide images (WSI). The *label* is defined on the level of *patients*, not on the level of pixels in a given WSI. Thus, the problems addressed by HAI are *weakly supervised problems*.

This is important to notice that there are various changes in this version but it follows the same steps.

++ These scripts are still under the development and please always use the final version of it ++

## How to use this repository:
To use this workflow, you need to modfiy specific experiement file based on your project. Experiment file is a text file and an example of it can be find this repository. For this file you need to fill the following options:

Input Variable name | Description
--- | --- 
-projectDetails | This is an optional string input. In this section you can write down some keywords about your experiment.| 
-dataDir_train | Path to the directory containing the normalized tiles. For example : ["K:\\TCGA-CRC-DX"]. <br/> This folder should contain a subfolder of tiles which can have one of the following names: <br/> {BLOCKS_NORM_MACENKO, BLOCKS_NORM_VAHADANE, BLOCKS_NORM_REINHARD or BLOCKS}. <br/>The clinical table and the slide table of this data set should be also stored in this folder. <br/>This is an example of the structure for this folder: <br/> K:\\TCGA-CRC-DX: <br/> { <br/> 1. BLOCKS_NORM_MACENKO <br/>2. TCGA-CRC-DX_CLINI.xlsx <br/>3. TCGA-CRC-DX_SLIDE.csv <br/> }
-dataDir_test | If you are planning to have external validation for your experiemnt, this varibal is the path to the directory containing the normalized tiles which will be used in external validation. This folder should have the same structure as the 'dataDir_train'.
-targetLabels | This is the list of targets which you want to analyze. The clinical data should have the values for these targets. For Example : ["isMSIH", "stage"].
-trainFull | If you are planning to do cross validation, this variable should be defined as False. If you want to use all the data to train and then use the external validation, then this variable should be defined as True.
-maxNumBlocks | This integer variable, defines the maximum number of tiles which will be used per slide. Since the number of extracted tiles per slide can vary alot, we use limited number of tiles per slide. For more detail, please ckeck the paper.
-epochs | This integer variable, defines the number of epochs for training. 
-batchSize |  This integer variable, defines the batch size for training. 
-modelName | This is a string variable which can be defined using one of the following neural network models. The script will download the pretrained weights for each of these models.<br/> {resnet, alexnet, vgg, squeezenet, densenet, inception, vit, efficient}
-opt | This is a string variable defining the name of optimizer to use for training. <br/> {"adam" or "sgd"}
-lr | This float variable defines the learning rate for the optimizer.
-gpuNo | If the computer has more than one gpu, this variable can be assigned to run the experiment on specified gpu.  

## Run training :

To start training, we use the Main.py script. The full path to the experiemnt file, should be used as an input variable in this script.
