# SWARM Learning For Histopathology Image Analysis

The objective of this repository is to replicate the Swarm Learning experiments described in [Saldanha et al., biorxiv, 2021](https://www.biorxiv.org/content/10.1101/2021.11.19.469139v1.full). This study demonstrates the feasibility of decentralized training of AI systems in computational pathology via Swarm Learning. The basic procedure was previously used for transcriptomics data in [Warnat-Herresthal et al., Nature 2021](https://rdcu.be/cA9XP). Similarly to this previous study, we use [HPE Swarm Learning](https://github.com/HewlettPackard/swarm-learning) as the core Swarm Learning package in our pipeline. In this repository, we describe the pipeline which integrates Swarm Learning with an end-to-end computational pathology workflow. The pathology image analysis workflow (single-center) was described [Ghaffari Laleh et al., biorxiv 2021](https://www.biorxiv.org/content/10.1101/2021.08.09.455633v1.full.pdf) as well as in several previous papers including [Kather et al., Nature Medicine 2019](https://dx.doi.org/10.1038/s41591-019-0462-y).

## Installation & Requirements

In general, the following is required in order to reproduce this experiment:
* Three physically separated computer systems (in this repository they will be referred to as *System A*, *System B*, and *System C*) with an internet connection
* These systems must be running Linux natively. We recommend to use newly created users with docker installed for all users. Running Linux in a virtual machine requires additional workarounds which are not described here. Here, we used Ubuntu 20.04.
* At each system, the user requires administrator privileges for some installation steps. We recommend to switch on sudo privileges for the current user like this:
    1. `sudo usermod -a -G sudo \<username>` where ‘username’ is the name of the current user. Be aware that this should be disabled after running the experiments for security reasons.
* The user at each system requires [Docker](https://hub.docker.com/) which can be installed like this:
    1. `sudo apt-get update`
    2. `sudo apt-get install docker-ce docker-ce-cli containerd.io`
* Each user must be part of a docker group. This can be achieved by running the following command-line script from a user account with admin privileges in each system:
    1. `sudo usermod -a -G docker \<username>`
* (optional) We recommend that each system has a CUDA-enabled GPU for faster training. Here, we propose a two-step approach with offline feature extraction and subsequently training the swarm network on these features, which speeds up training. This also allows training on computers without a GPU in reasonable time

## Example Data Set

A small example dataset has been provided along with this repository. We extracted four subsets from the [TCGA colorectal (CRC)](https://gdc.cancer.gov/about-data/publications/coadread_2012) cohort. The subsets are taken from four [contributing sites](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes) in TCGA: CM (Memorial Sloan Kettering Cancer Center), D5 (Greater Poland Cancer Center), G4 (Roswell Park) and A6 (Christiana Healthcare). Provided three systems are being used as suggested above, each system can be allocated a unique subcohort, with the remaining subcohort being used as an external test dataset. Because these datasets are much smaller than the ones in our study, the performance can vary markedly between multiple runs (unlike with large cohorts, where you usually get very similar results in multiple runs). 

These datasets have been preprocessed according to [“The Aachen Protocol for Deep Learning Histopathology: A hands-on guide for data preprocessing”](https://zenodo.org/record/3694994#.Yea3I9DMIu) The whole slide images (WSIs) were [tessellated (without any annotations), normalized](https://github.com/KatherLab/preProcessing) and feature vectors were extracted for all slides by using an off-the-shelf resnet model. The original WSIs are available at the [GDC Data Portal](https://portal.gdc.cancer.gov/) (COAD and READ, together referred to as CRC). In this example, the features are already extracted and are saved in the folders "System A", "System B" and "System C" in this repository. 

When using your own data, you can tessellate the WSIs using the above-mentioned references and extract features using the scripts in this repository. 

## System  Preparation

Note: unless otherwise stated, the following must be done for each of the three systems.
1. Clone this Github repository to each System
2. Unzip the Dataset into the folder ***SWARM/System A/data*** for all systems
3. Change Hyperparameters:
    1. On System A, get the IP address (open a terminal, run the command:`hostname  -I | cut -f1 -d `). 
    2. On System B and System C open the [sl-node](System%20B/sl-node.sh) and [sn-node](System%20C/sn-node-sentinel.sh) with an editor and insert the previously noted IP-address from System A in the predefined line (eg: `system_A_ip=137.226.23.146`).   
    3. (Optional) the target label (prediction target) can be changed inside the [experiment file](System%20A/MODEL/exp_A.txt). The user has to provide the target name that the model will train on on all 3 Systems. In our case, we train on microsatellite instability (MSI) status, the target is called "isMSIH" with two levels: "MSIH" and "nonMSIH".
 
5. Setting up docker in all the systems:
    1. Login to the docker using the terminal type: `docker login hub.myenterpriselicense.hpe.com -u <HPE-PASSPORT-EMAIL> -p hpe_eval`
    2. Enable docker content trust `export DOCKER_CONTENT_TRUST=1`
    3. Create a docker image with the name `pyt-cv2` using the Dockerfile on all systems:
        * open terminal in docker folder
        * `docker build -t pyt-cv2 .`
6. Connect systems via a passwordless ssh and create a docker image:
    1. (Optional) passwordless SSH (note: if not performed, disruptive password inputs will otherwise be required at multiple stages of the experiment)
       Has to be done on Systems B and C
        *  open a terminal and run `ssh-keygen`
        *  run `cat ~/.ssh/id_rsa.pub`
        *  run `ssh <linux username System A>@<IP of System A>`
        *  run `mkdir ~/.ssh`
        *  run `cat > ~/.`
        *  run `cat >> ~/.ssh/authorized_keys`
 
## Run Experiment

1. (Only on System A) Run the swarm learning setup
    1. Open a terminal in "SWARM\System A\swarm-learning\bin"
    2. `bash run-apls`  
    An example of the expected output::
    ![alt text](https://github.com/KatherLab/SWARM/blob/main/run_apls.png?raw=true)
    4. Upload the HPE license key:
        1. open the following website in your browser: `https://<ip>:5814/autopass/login_input` however substitute the ip with System As ip-adress(eg. `https://137.226.23.146:5814/autopass/login_input`)
        2. Use the default settings user_name: *admin*, and password: *password* and change the password as prompted
        3. Perform the steps in the following image:
        ![alt text](https://github.com/KatherLab/SWARM/blob/main/login.png?raw=true)
        4. A message should appear in the browser that the license key has been uploaded successfully.
        5. Do not close the terminal and the browser window.
2. (Only on System A) In a new terminal, Start the SPIRE server .sh file in “SWARM\System   A\”spire-server.sh
    1. Go to System A/
    2. `sh spire-server.sh`
    3. Wait until the last lines of the output appears as follows:
    ![alt text](https://github.com/KatherLab/SWARM/blob/main/spire-server.png?raw=true)
3. (Only on System A ) In a new terminal Run the SN Node:
    1. Go to System A/
    2. `sh sn-node-sentinal.sh`
    3. Wait until the port appear similar to the following:
    ![alt text](https://github.com/KatherLab/SWARM/blob/main/sn-node.png?raw=true) 
4. Run the sn-node.sh file in the other two systems:
    1. Go to System #/     #do so for B on *System B* and C on *System C*
    2. `sh sn-node.sh`
    3. Wait until the output looks similar to the screenshot above.
5. Run sl-node in all three systems
    1. Go to System #/ #do so for all three systems
    2. `sh sl-node.sh`
    3. This will initialize the training of the model. The expect output is as follows:
    ![alt text](https://github.com/KatherLab/SWARM/blob/main/sl-node.png?raw=true)
6. As soon as all systems are done, the training is will finish. The final, trained model will be saved in SWARM/System A/MODEL/saved_model/ as a .pkl file.

## Troubleshooting

* In the event that when starting a node, any of the desired messages as shown in the screenshots above do not appear, starting the node again or redoing the whole process may resolve the issue.
* Further information regarding the use of HPE Swarm Learning can be found in the documentation section of the following repository: [HPE Swarm Learning](https://github.com/HewlettPackard/swarm-learning). Issues regarding the HPE package should be posted there and are usually responded to by the HPE team. 

## License

All data and source codes in this repository are released under the MIT license:

`Copyright 2021-2022
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.`



