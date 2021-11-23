##!/usr/bin/env python3
"""
Created on Mon Sep 27 13:13:41 2021

@author: oliver
"""

import utils.utils as utils
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from utils.data_utils import ConcatCohorts_Classic, GetTiles
from utils.core_utils import Validate_model_Classic
from eval.eval_Classic import CalculatePatientWiseAUC
import torch
import os
import random
from sklearn import preprocessing
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy
from dataclasses import dataclass
from  ROC import ROC_custom

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--adressExp', type = str, default = r"path-to-deploy-expirement_file/deploy.txt", help = 'Adress to the experiment File')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class mnistNet(nn.Module):
    def __init__(self):
        super(mnistNet, self).__init__()
        self.dense = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 2)
    def forward(self, x):
        #x = torch.flatten(x, 1)        
        x = self.dense(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.dense1(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.dense2(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.dense3(x)
        x = F.relu(x)
        #x = self.dropout(x)
        x = self.dense4(x)
        output = x
        return output
##############################################################################

if __name__ == '__main__':
    args = utils.ReadExperimentFile(args, deploy = True)
    args.useCsv = True

    torch.cuda.set_device(args.gpuNo)
    random.seed(args.seed)

    targetLabels = args.target_labels
    args.feature_extract = False
    for i in range(5):
        targetLabel = args.target_labels[0]
        i=i+5
        #path to save ressults
        path_to_save_results = r"path-to-same-results/Expirement_"+str(i)+"/target/"
        os.makedirs(path_to_save_results,exist_ok = True)
        #path to get features and lables of deployment(validation) cohort
        projectFolder = r"/path-to-validation-cohort-features-lables"
        deploy_name = projectFolder.split('y/')
        deploy_name=deploy_name[1].split('_')
        reportFile  = open(os.path.join(projectFolder,'Report.txt'), 'a', encoding="utf-8")
        reportFile.write('**********************************************************************'+ '\n')   
        targetLabels = args.target_labels
        print('\nLoad the DataSet...')                      
        patientsList, labelsList, slidesList, clinicalTableList, slideTableList = ConcatCohorts_Classic(imagesPath = args.datadir_test, cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
        	                                        label = targetLabel, reportFile = reportFile)
        print('\nLOAD THE DATASET FOR TRAINING...\n')  
        labelsList = utils.CheckForTargetType(labelsList)
        le = preprocessing.LabelEncoder()
        labelsList = le.fit_transform(labelsList)
        args.num_classes = len(set(labelsList))
        args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        
        args.result = os.path.join(projectFolder, 'RESULTS')
        os.makedirs(args.result, exist_ok = True)
        features_file_name = os.path.join(projectFolder, 'feature_lables/test_features.npy')
	
	     # Check if the file  having featurs and lables already exists if not extract the features
        if os.path.isfile(features_file_name):
            feature_lable_path = os.path.join(projectFolder, 'feature_lables')
            csv_path = os.path.join(projectFolder, 'SPLITS/FULL_TEST.csv')
            test_data = pd.read_csv(csv_path)
            #print(test_data.columns)
            test_x = list(test_data['X'])
            test_y = list(test_data['y'])
            test_pid = list(test_data['patientID'])
            #print('##############')
        else:
            args.split_dir = os.path.join(projectFolder, 'SPLITS')
            os.makedirs(args.split_dir, exist_ok = True)
            args.feature_label_dir = os.path.join(projectFolder, 'feature_lables')
            os.makedirs(args.feature_label_dir, exist_ok = True)
            print('GENERATE NEW TILES')    
            test_pid = []
            test_x = []
            test_y = []          
            patientID = np.array(patientsList)
            test_data = GetTiles(patients = patientID, labels = labelsList, imgsList = slidesList, label= targetLabel,
        	                      slideTableList = slideTableList, maxBlockNum = args.maxBlockNum, test = True) 
            test_x = list(test_data['tileAd'])
            test_y = list(test_data[targetLabel])
            test_pid = list(test_data['patientID'])               
            df = pd.DataFrame(list(zip(test_pid,test_x, test_y)), columns =['patientID','X', 'y'])
            df.to_csv(os.path.join(args.split_dir, 'FULL_TEST'+ '.csv'), index = False)
            path_train = os.path.join(args.split_dir, 'FULL_TEST'+ '.csv')
            
            @dataclass
            class FeatureExtraction():
            	spreadsheet_path: str
            	deep_med_model: str = None
                   
            	model = models.resnet18(pretrained=True).to(device)
            	layer = model._modules.get('avgpool')
            	model.eval()
            	scaler = transforms.Scale((224, 224))
            	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            	                         std=[0.229, 0.224, 0.225])
            	to_tensor = transforms.ToTensor()
                   
                   
            	def get_feature_vectors(self, image_path: str) -> torch.Tensor:
            	    img = Image.open(image_path)
            	    if img.mode != "RGB":
            	        img = img.convert("RGB")
            	    t_img = Variable(self.normalize(self.to_tensor(self.scaler(img)).to(device)).unsqueeze(0)) ###
            	    embedding = torch.zeros(512)
            	    def copy_data(m, i, o):
            	        embedding.copy_(o.data.reshape(o.data.size(1)))
            	    h = self.layer.register_forward_hook(copy_data)
            	    self.model(t_img)
            	    h.remove()
            	    return embedding
            	   
            	    def get_label(x):
            	        i_image = df[df["X"] == x].index.values[0]
            	        label = df["y"][i_image]
            	        if label.isnumeric():
            	            return torch.tensor(float(df["y"][i_image]))
            	        else:
            	            unique_labels = sorted(df["y"].unique())
            	            label_dict = dict()
            	            for i in range(len(unique_labels)):
            	                label_dict[unique_labels[i]] = i 
            	            return float(label_dict[label])
            	   
            	    df = pd.read_csv(self.spreadsheet_path, dtype=str)
            	    all_features = []
            	    labels = []
            	    image_names = []
            	   
            	    for im_name in df["X"]:
            	        image_path = im_name
            	        image_vector = self.get_feature_vectors(image_path)
            	        label = get_label(im_name)
            	        all_features.append(image_vector)
            	        labels.append(label)
            	        image_names.append(image_path)
                   
            	    all_features = torch.stack((all_features), dim=0)
            	    torch_labels = torch.stack((labels), dim=0)
            	    print(torch_labels.shape)
            	    np_all_features = all_features.numpy()
            	    labels = torch_labels.reshape(-1).t()
            	    np_labels = labels.numpy()
            	    return np_all_features, np_labels
               
            
            
            fc_train = FeatureExtraction(path_train)
            np_train_feature_vectors, np_tarin_labels = fc_train.extract_and_save_feature_vectors()
            path_train_feature_save = os.path.join(args.feature_label_dir, 'test_features.npy')
            path_train_lable_save = os.path.join(args.feature_label_dir, 'test_lable.npy')
            numpy.save(path_train_feature_save, np_train_feature_vectors)
            numpy.save(path_train_lable_save, np_tarin_labels)
            feature_lable_path = args.feature_label_dir

        def loadData(feature_label_path):
            # load data from npz format to numpy
            xTest, yTest = np.load(os.path.join(feature_label_path, 'test_features.npy')), np.load(os.path.join(feature_label_path, 'test_lable.npy'))      
            # transform numpy to torch.Tensor
            xTest, yTest = map(torch.tensor, (xTest.astype(np.float32), yTest.astype(np.int_)))    
            # convert torch.Tensor to a dataset
            yTest = yTest.type(torch.LongTensor)
            testDs = torch.utils.data.TensorDataset(xTest,yTest)
            return  testDs

        resultlists=[]       
        batchSz = args.batch_size
        testDs = loadData(feature_lable_path)    
        model_ft = mnistNet().to(device)
        print(model_ft)
        main_folder_path = r"path-to-save-roc"
        main_model_path = os.listdir(main_folder_path)
        
        for mainpath in main_model_path:
            model_folder_path = os.path.join(main_folder_path,str(mainpath))
            model_path = os.listdir(model_folder_path)
            print(model_path)
            for path in model_path:
            	print(path)
            	full_path = os.path.join(model_folder_path,str(path))
            	model_ft.load_state_dict(torch.load(full_path))
            	print(model_ft)
            	testLoader = torch.utils.data.DataLoader(testDs, batch_size=batchSz, shuffle =False)
            	criterion = nn.CrossEntropyLoss()
            	print('START DEPLOYING...')
            	print('')
            	epoch_loss, epoch_acc, predList  = Validate_model_Classic(model_ft, testLoader, criterion)
            	scores = {}
            	for index, key in enumerate(list(args.target_labelDict.keys())):
            	    scores[key] = []
            	    for item in predList:
            	        scores[key].append(item[index])
            	
            	scores = pd.DataFrame.from_dict(scores)
            	
            	df = pd.DataFrame(list(zip(test_pid, test_x, test_y)), columns =['patientID', 'X', 'y'])
            	df = pd.concat([df, scores], axis=1)
            	   
            	save_path= os.path.join(path_to_save_results , mainpath)	
            	    ##########################################################################################################
            	if not os.path.isdir(save_path):
            	    os.mkdir(save_path)#create path 
            	df.to_csv(os.path.join(save_path,path+'TEST_RESULT_FULL.csv'), index = False)   #savepath 
            	CalculatePatientWiseAUC(os.path.join(save_path,path+'TEST_RESULT_FULL.csv'),
            	                            list(set(test_pid)), args.target_labelDict, save_path ,  counter = 'FULL')
            	path2 = os.path.join(save_path,path+'TEST_RESULT_FULL.csvTEST_RESULT_PATIENT_SCORES_FULL.csv')
            	print(path2)
            	patient_count = path.split('saved_model')
            	patient_count= patient_count[1].split('.p')
            	if patient_count[0] == '0.99':
            	    title= "ROC of "+str(mainpath) + " with all patients tested on "+ deploy_name[0]
            	else:
            	    title= "ROC of "+str(mainpath) + " with "+patient_count[0]+" patients tested on "+deploy_name[0]
            	print(title)
            	savepath= str(path)
            	auroc_val,ci1,ci2,p_value = ROC_custom(path2,title,savepath)
            	
            	resultlists.append([mainpath,path,auroc_val,ci1,ci2,p_value])
            	print('#######################')
            df_all=pd.DataFrame(resultlists,columns=["Cohort","Modelname","Auroc","lowerCI","upperCI","p-value"])
            full_path_csv = path_to_save_results + "Experimentresults.csv"
            print(full_path_csv)
            df_all.to_csv(full_path_csv)
            
        
        
        
        
        
        
        
        
        
        
        
        
 
