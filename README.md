# SWARM Learning For Histopathology Image Analysis

The objective of this repository is to replicate a decentralized training of multiple cohorts in multiple centers as stated in the following paper: :exclamation:insert link to paper for swarm learning:exclamation:. It is based on workflows that were previously described in [Warnat-Herresthal et al., Nature 2021](https://rdcu.be/cA9XP) and [Laleh, N. G. et al., Gastroenterology 2020](https://www.biorxiv.org/content/10.1101/2021.08.09.455633v1.full.pdf). 

## Installation & Requirements:
In general, the following is required in order to reproduce this experiment:
* Three physically separated computer systems (in this repository they will be referred to as *System A*, *System B*, and *System C*, where System A will serve as the host)
* These systems must be running Linux through newly created users with docker installed for all users.
* All three users require sudo privileges. This can be achieved by running the following command-line script from a user account with admin privileges in each system:
    1. 'sudo usermod -a -G sudo \<username>'
    
    Where ‘username’ is the name of the users created as previously stated.
* All three users require Docker. This can be installed individually using the following command-line script:
    1. sudo apt-get update
    2. sudo apt-get install docker-ce docker-ce-cli containerd.io
* All three users require to be a part of docker group. This can be achieved by running the following command-line script from a user account with admin privileges in each system:
    1. 'sudo usermod -a -G docker \<username>'
* (optional) GPU for faster training 

## Provided Data: 

A small example dataset has been provided along with this repository. The provided whole slide images (WSIs) are tessellated, normalized and have had feature vectors generated. This data has been preprocessed according to [“The Aachen Protocol for Deep Learning Histopathology](https://zenodo.org/record/3694994#.Yea3I9DMIu): A hands-on guide for data preprocessing”
* The four subcohorts of the example dataset can be downloaded at exclamation:link (still missing)
* Provided three systems are being used as instructed above, each system can be allocated a unique subcohort, with the remaining subcohort being used as an external test dataset.

When using your own data, it is strongly recommended to preprocess the data according to the Aachen Protocol for Deep Learning Histopathology. This can be done using the following code: (https://github.com/KatherLab/preProcessing)


## System  Preparation:
Note: unless otherwise stated, the following must be done for all systems! 
1. Clone this Github repository to each System
2. Unzip the Dataset into the folder ***SWARM/System A/data*** for all systems
3. Change Hyperparameters:
    1. On System A, get the IP-address (open a terminal, run the command:`hostname  -I | cut -f1 -d' '` )
    2. On System B and System C open  the [sl-node](System%20B/sl-node.sh) and [sn-node](System%20C/sn-node-sentinel.sh) with an editor and insert the previously noted IP-address from System A  in the predefined line (eg: `system_A_ip=137.226.23.146`). 
    3. (Optional) the target label can be changed inside the [experiment file](System%20A/MODEL/exp_A.txt). One has to provide the target name that the model will train on(e.g. 'isMSIH') on all 3 Systems:  
 
5. Setting up docker in all the systems:
    1. Login to the docker using the terminal type: `docker login hub.myenterpriselicense.hpe.com -u <HPE-PASSPORT-EMAIL> -p hpe_eval`
    2. Enable docker content trust `export DOCKER_CONTENT_TRUST=1`
    3. Create a docker image with the name ‘pyt-cv2’ using the Dockerfile on all systems:
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
    4. Upload the license key:
        1. open the following website in your browser: `https://<ip>:5814/autopass/login_input` however substitute the ip with System A's ip-adress(eg. `https://137.226.23.146:5814/autopass/login_input`)
        2. Use the default settings user_name: *admin*, and password: *password* and change the password as prompted
        3. Perform the steps in the following image:
        ![alt text](https://github.com/KatherLab/SWARM/blob/main/login.png?raw=true)
        4. A message should appear in the browser that the license key has been uploaded successfully.
        5. Do not close the terminal and the browser window.
2. (Only on System A) In a new terminal, Start the spire-server .sh file in “SWARM\System   A\”spire-server.sh
    1. Go to 'System A/'
    2. `sh spire-server.sh`
    3. Wait until the last lines of the output appears as follows:
    ![alt text](https://github.com/KatherLab/SWARM/blob/main/spire-server.png?raw=true)
3. (Only on System A ) In a new terminal Run the SN Node:
    1. Go to 'System A/'
    2. `sh sn-node-sentinal.sh`
    3. Wait until the port appear similar to the following:
    ![alt text](https://github.com/KatherLab/SWARM/blob/main/sn-node.png?raw=true) 
4. Run the sn-node.sh file in the other two systems:
    1. Go to 'System #/'     #do so for B on *System B* and C on *System C*
    2. `sh sn-node.sh`
    3. Wait until the output looks similar to the screenshot above.
5. Run sl-node in all three systems
    1. Go to 'System #/' #do so for all three systems
    2. `sh sl-node.sh`
    3. This will initialize the training of the model. The expect output is as follows:
    ![alt text](https://github.com/KatherLab/SWARM/blob/main/sl-node.png?raw=true)
6. . As soon as all systems are done, the training is will finish. The final, trained model will be saved in SWARM/System A/MODEL/saved_model/ as a .pkl file.

**Additionally**:
* In the event that when starting a node, any of the desired messages as shown in the screenshots above do not appear, starting the node again or redoing the whole process may resolve the issue.
* Pre-print of the scientific paper about **Swarm learning for decentralized artificial intelligence in cancer histopathology** could be found in the following link: (https://www.biorxiv.org/content/10.1101/2021.11.19.469139v1.full)
* Further information regarding the use of HPE SWARM learning can be found in the documentation section of the following repository: (https://github.com/HewlettPackard/swarm-learning)
