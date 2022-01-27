# SWARM Learning For Histopathology Image Analysis

The objective of this repository is to replicate a decentralized training multiple cohorts in multiple centers as stated in the following paper : :exclamation:insert link to paper for swarm learning:exclamation:. It is based on workflows which were previously described in [Warnat-Herresthal et al., Nature 2021](https://rdcu.be/cA9XP) and [Laleh, N. G. et al., Gastroenterology 2020](https://www.biorxiv.org/content/10.1101/2021.08.09.455633v1.full.pdf). 

## Installation & Requirements:
In general, the following requirements are needed to reproduce this experiment: 
* HPE Account for swarm learning license:  https://myenterpriselicense.hpe.com/cwp-ui/auth/login
* three physically separated computer systems (in this repository they will be refered as *System A*,*System B* and *System C*, where System A will serve as the host)

**Furthermore** each system requires:
*  **Linux-Distribution**
* Administration rights
* the linux account names must be the same on all systems
* (optional) GPU for faster training 




## Provided Data: 

For this repository, a small example dataset is provided, where the WSI are given as tesselated and normalized tiles and also as feature vectors:
* Download four subcohorts from  [given link]() :exclamation:link still missing:exclamation:
* Provide one cohort for each System, the fourth cohort can be used as an external test set afterwards

To use one's own data, it is recommended to  use [this preprocessing code](https://github.com/KatherLab/preProcessing) and provide the dataset according to: [“The Aachen Protocol for Deep Learning Histopathology](https://zenodo.org/record/3694994#.Yea3I9DMIu): A hands-on guide for data preprocessing”

## System  Preparation:
Has to be done equally on each System if not said otherwhise! 
1. clone this Github repository on each System
2. Unzip the Dataset into the respective System into the folder ***SWARM/System A/swarm_setup_training/data*** for all systems
3. Change Hyperparameters:
    1. On System A get the ip adress (open a terminal, run the command:`hostname  -I | cut -f1 -d' '` )
    2. On System B and System C open  the [sl-node](System%20B/swarm_setup_training/sl-node.sh) and [sn-node](System%20C/swarm_setup_training/sn-node-sentinel.sh) with an editor and insert the previously noted ip-adress from System A  in the predefined line (eg: `system_A_ip=137.226.23.146`). 
    3. (Optional) the target label can be changed inside the [experiment file](System%20A/swarm_setup_training/MODEL/expirement_file.txt). One has to provide the target name that the model will train on(e.g. 'isMSIH') on all 3 Systems:  
   
4. Connect Computers via ssh:
    1. On each system  run in a command line  `sudo usermod -a -G docker <linux username System>` 
    2. Create a docker image with the name ‘pyt-cv2’ using the Dockerfile on all systems:
        * open terminal in docker folder
        * `docker build -t pyt-cv2 .`
    3. (Optional) passwordless SSH:\
       Has to be done on the Systems B and C
        *  open a terminal and run `ssh-keygen`
        *  run `cat ~/.ssh/id_rsa.pub`
        *  run `ssh <linux username System A>@<IP of System A>`
        *  run `mkdir ~/.ssh`
        *  run `cat > ~/.`
        *  run `cat >> ~/.ssh/authorized_keys`
 
## Run Experiment

1. (Only on System A) Run the swarm learning setup
    1. open a terminal in "SWARM\System A\swarm_setup_training\swarm-learning\bin"
    2. `bash run-apls`  
    3. upload the license key:
        1. open the following website in your browser: `https://<ip>:5814/autopass/login_input` however substitute the ip with the System A's ip-adress(eg. `https://137.226.23.146:5814/autopass/login_input`)
        2. Use the default settings user_name: *admin*, and password: *password* and change the password as requested
        3. follow the steps in the following image:
        ![alt text](https://github.com/KatherLab/SWARM/blob/main/login.png?raw=true)
    5. A message should appear in the browser that the license key has been uploaded successfully.
2. (Only on System A) Start the spire-server .sh file in “SWARM\System   A\swarm_setup_training”spire-server.sh
    1. go to 'System A/swarm_setup_training/'
    2. `sh spire-server.sh`
3. (Only on System A ) Run the SN Node:
    1. go to 'System A/swarm_setup_training/'
    2. `sh sn-node-sentinal.sh`
    3. wait until port is ready :exclamation: clear message:exclamation: 
4. Run the sn-node.sh file in the other two systems:
    1. go to 'System #/swarm_setup_training/'     #do so for A on *System A* and B on *System B*
    2. `sh sn-node-sentinal.sh`
    3. wait until ports are ready :exclamation: clear message:exclamation: 
5. Run sl-node in all three systems
    1. go to 'System #/swarm_setup_training/' #do so for all three systems
    2. `sh sl-node.sh`
6. As soon as the required number of systems are done, the training is finished

**Additionaly**:

* It might happen  that after starting a node, the desired message doesn't appear. It helps many times to start the node again or redo the whole process
