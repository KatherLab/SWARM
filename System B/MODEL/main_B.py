import warnings
import argparse
import torch
from utils.data_utils import ConcatCohorts_Classic, DatasetLoader_Classic, LoadTrainTestFromFolders, GetTiles
import utils.utils as utils
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from train_function import doTrainBatch, test
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from dataclasses import dataclass
import torchvision.models as models
import pandas as pd
from swarm import SwarmCallback

#Getting all the arguments from the experement file
parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
exp_name = 'saved_model'
parser.add_argument('--adressExp', type = str, default = "./exp_B.txt", help = 'Adress to the experiment File')
args = parser.parse_args() 

#setting the syn interval to the swarm learning
swSyncInterval = 4

#Check for the CUDA and if available assign it
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
warnings.filterwarnings("ignore")
args = utils.ReadExperimentFile(args)
args.useCsv = True

# Define the model to be used
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

#Reading all the arguments in the experiment file
stats_total = {}
stats_df = pd.DataFrame()
args.target_label = args.target_labels[0]
targetLabel=args.target_labels

#path to the features and lables
feature_and_label_path = r"./Data_features/"+str(targetLabel[0])+"/Features/Data_set_B_"+str(targetLabel[0])+"/feature_lables"
print('###############################################################')

# Check if the file  having featurs and lables already exists if not extract the features
if os.path.exists(feature_and_label_path):
    #raise NameError('THis PROJECT IS ALREADY EXISTS!')
    print('THis PROJECT IS ALREADY EXISTS!')
    print()
    print('Features are taken from path:'+ feature_and_label_path)
    feature_lable_dir = feature_and_label_path
    feature_lable_path = feature_and_label_path
else:
    os.mkdir(args.projectFolder)
    # Create a report file to put in all the enteries
    reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
    reportFile.write('**********************************************************************'+ '\n')
    reportFile.write(str(args))
    reportFile.write('\n' + '**********************************************************************'+ '\n')
   
       
    patientsList, labelsList, slidesList, clinicalTableList, slideTableList = ConcatCohorts_Classic(imagesPath = args.datadir_train, cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                    label = args.target_label, reportFile = reportFile)
    print('\nLOAD THE DATASET FOR TRAINING...\n')  
    final_patient_list = patientsList
    values, counts = np.unique(labelsList, return_counts=True)
   
    le = preprocessing.LabelEncoder()
    labelsList = le.fit_transform(labelsList)
   
    args.num_classes = len(set(labelsList))
    args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        
     
    utils.Summarize_Classic(args, list(labelsList), reportFile)
           
    print(patientsList)
    print('IT IS A train test split of 90 10!')
    print('USE PRE SELECTED TILES')
    patientID = np.array(patientsList)
    labels = np.array(labelsList)
    print(patientID)
    args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
    os.makedirs(args.split_dir, exist_ok = True)
    args.feature_label_dir = os.path.join(args.projectFolder, 'feature_lables')
    os.makedirs(args.feature_label_dir, exist_ok = True)
     
    counter = 0
   
   
    trainData_patientID , testData_patientID, trainData_labels , testData_Labels  = train_test_split(patientID, labels, test_size=0.001)
    print(testData_patientID)
    train_index=[]
    test_index=[]
    for i in range(len(trainData_patientID)):  
        a =  patientID.tolist().index(trainData_patientID[i])
        train_index.append(a)
    print(train_index)
    for j in range(len(testData_patientID)):  
        b =  patientID.tolist().index(testData_patientID[j])
        test_index.append(b)
    print(test_index)
     
   
    #print(kf.split(patientID, labels).shape)
    #train_index, test_index = .split(patientID, labels)
    print('GENERATE NEW TILES')
    #train_index = [i for i in train_index if i not in val_index]
   
    counter = 0
    print('\nLOAD TRAIN DATASET\n')
       
    train_data = GetTiles(patients = trainData_patientID, labels = trainData_labels, imgsList = slidesList, label = args.target_label, slideTableList = slideTableList, maxBlockNum = args.maxBlockNum, test = False, seed = args.seed)
   
    train_x = list(train_data['tileAd'])
    train_y = list(train_data[args.target_label])
       
    df = pd.DataFrame(list(zip(train_x, train_y)), columns =['X', 'y'])
    print(df)
   
    df.to_csv(os.path.join(args.split_dir, 'SPLIT_TRAIN_' + str(counter)+ '.csv'), index = False)
    path_train = os.path.join(args.split_dir, 'SPLIT_TRAIN_' + str(counter)+ '.csv')
   
    #p_df = path_train[0] + "\\" + path_train[1]
   
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
            """
            Extracts the feature vector of the given image from the avgpool layer of ResNet-18 and
            returns the feature vector as a tensor
           
           
            Parameters:
            image_path (str): The absolute path to the image itself
           
            Returns:
            embedding (torch.Tensor)
           
            """
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
   
        def extract_and_save_feature_vectors(self):
            """
            Extracts and saves the feature vectors of all images in the given spreadsheet and
            saves the feature vector and the labels of images as a tensor
           
            Parameters:
            image_path (str): The absolute path to the image itself
            """
            if self.deep_med_model != None:
                self.model = load_learner(deep_med_model).model.to(device)
           
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
            # numpy.save("./feature_vectors_" + self.name_project + ".npy", np_all_features)
            # numpy.save("./labels_" + self.name_project + ".npy", np_labels)
           
   
   
    fc_train = FeatureExtraction(path_train)
    np_train_feature_vectors, np_tarin_labels = fc_train.extract_and_save_feature_vectors()
    #print(np_train_feature_vectors, np_tarin_labels)
    path_train_feature_save = os.path.join(args.feature_label_dir, 'train_features.npy')
    path_train_lable_save = os.path.join(args.feature_label_dir, 'train_lable.npy')
    numpy.save(path_train_feature_save, np_train_feature_vectors)
    numpy.save(path_train_lable_save, np_tarin_labels)
       
    print('LOAD TEST DATASET')  
    test_data = GetTiles(patients = testData_patientID, labels = testData_Labels, imgsList = slidesList, label = args.target_label,
                          slideTableList = slideTableList, maxBlockNum = args.maxBlockNum, test = True, seed = args.seed)  
   
    test_x = list(test_data['tileAd'])
    test_y = list(test_data[args.target_label])
    test_pid = list(test_data['patientID'])
       
    df = pd.DataFrame(list(zip(test_pid, test_x, test_y)), columns = ['pid', 'X', 'y'])
    #print(df)
    df.to_csv(os.path.join(args.split_dir, 'SPLIT_TEST_' + str(counter) + '.csv'), index = False)
   
    path_test = os.path.join(args.split_dir, 'SPLIT_TEST_' + str(counter) + '.csv')
    fc_test = FeatureExtraction(path_test)
    np_test_feature_vectors, np_test_labels = fc_test.extract_and_save_feature_vectors()
   
    #print(np_test_feature_vectors, np_test_labels)
    path_test_feature_save = os.path.join(args.feature_label_dir, 'test_features.npy')
    path_test_lable_save = os.path.join(args.feature_label_dir, 'test_lable.npy')
    np.save(path_test_feature_save, np_train_feature_vectors)
    np.save(path_test_lable_save, np_tarin_labels)
    feature_lable_path = args.feature_label_dir

def loadData(feature_label_path):
    # load data from npz format to numpy
    xTrain, yTrain =np.load(os.path.join(feature_label_path, 'train_features.npy')), np.load(os.path.join(feature_label_path, 'train_lable.npy'))
    xTest, yTest = np.load(os.path.join(feature_label_path, 'test_features.npy')), np.load(os.path.join(feature_label_path, 'test_lable.npy'))
    #xTrain, xTest = xTrain / 255.0, xTest / 255.0        
       
    # transform numpy to torch.Tensor
    xTrain, yTrain, xTest, yTest = map(torch.tensor, (xTrain.astype(np.float32),
                                                      yTrain.astype(np.int_),
                                                      xTest.astype(np.float32),
                                                      yTest.astype(np.int_)))    
    # convert torch.Tensor to a dataset
    yTrain = yTrain.type(torch.LongTensor)
    yTest = yTest.type(torch.LongTensor)
    trainDs = torch.utils.data.TensorDataset(xTrain,yTrain)
    testDs = torch.utils.data.TensorDataset(xTest,yTest)
    return trainDs, testDs

batchSz = args.batch_size
trainDs, testDs = loadData(feature_lable_path)
model_ft = mnistNet().to(device)

print(model_ft)
print('\nINIT OPTIMIZER ...', end = ' ')
optimizer = utils.get_optim(model_ft, args, params = False)
print('DONE!')

criterion = nn.CrossEntropyLoss()
trainLoader = torch.utils.data.DataLoader(trainDs, batch_size=batchSz, shuffle =True)
testLoader = torch.utils.data.DataLoader(testDs, batch_size=batchSz, shuffle =False)

print('\nSTART TRAINING ...', end = ' ')

# get the bath size from the expirement file
max_epochs = args.max_epochs
batchSz = 124
model_name = 'test-example'
swarmCallback = None
default_max_epochs = 4

# provide a sawrm call back
swarmCallback = SwarmCallback(sync_interval=swSyncInterval,
                          min_peers=2,
                          val_data=testLoader,
                          val_batch_size=batchSz,
                          model_name=model_name,
                          #node_weightage = 63,
                          model=model_ft)
                          
# initalize swarmCallback and do first sync
swarmCallback.on_train_begin()

for epoch in range(1, max_epochs + 1):
        trainedModel =doTrainBatch(model = model_ft,device = device, trainLoader = trainLoader, optimizer = optimizer, epoch = epoch,swarmCallback=swarmCallback)      
        test(model = trainedModel,device = device, testLoader = testLoader)
        swarmCallback.on_epoch_end(epoch)
swarmCallback.on_epoch_end(epoch)

print('DONE!')

model_path = os.path.join(exp_name,'saved_model_with_'+str(targetLabel[0])+'.pkl')
print(model_path)
torch.save(trainedModel.state_dict(), model_path)
