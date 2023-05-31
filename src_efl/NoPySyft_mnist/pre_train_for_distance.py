import argparse

import math
import pandas as pd

import os
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch import nn
from torchsummary import summary

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from client_node import ClientNode
from mnist_model import MLP2
#from model import CNN2 #, SimpleCNN # for cifar10, cifar100
from data_loader import Data_Loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class pre_train:
    def __init__(self, n_clients, global_rounds, batch_size, learning_rate, lr_decay, z_dir, w_dir,
        dataset, pre_trained_w_file = None, partition_data_file = None, distance_metric = 'euclidean', p = 1):

        self.partition_data_file = partition_data_file
        self.global_rounds = global_rounds
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        
        self.dataset = dataset
        self.data_loader = Data_Loader(dataset, batch_size)#Data_Loader_EfficientNet_Cifar100(batch_size)
        self.train_data = None
        self.test_data = None
        self.data_size = [] # for aggregate model
        
        self.z_dir = z_dir
        self.w_dir = w_dir
        self.partition_data_file = partition_data_file
        self.distance_metric = distance_metric
        self.p = p

        #self.model = CNN2(hidden_dims = [256,128], output_dim = 100).to(device) #cifar100, cifar10
        #summary(self.model, input_size=(3, 32, 32)) #channel, img_size, img_size
        #self.model_name = "CNN2"

        #self.model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=100).to(device) #cifar100, cifar10
        #summary(self.model, input_size=(3, 32, 32)) #channel, img_size, img_size
        #self.model_name = "sCNN"

        self.model =  MLP2().to(device)#CNN1().to(device) #MLP2().to(device) #MNIST().to(device) #CNNModel().to(device) MNIST FEMNIST
        summary(self.model, (1, 28, 28))
        self.model_name = "MLP2"

        if (pre_trained_w_file != None):
            print("Loading weight from " + pre_trained_w_file)
            self.model.load_state_dict(torch.load(self.w_dir + pre_trained_w_file)) 

        self.clients = self.generate_clients(n_clients)
        
    
    def get_distance_and_weight_files(self):
        
        
        if (self.partition_data_file != None):
            print("Loading partition_data_file: ", self.partition_data_file)
            self.train_data = self.data_loader.prepare_in_batch_given_partition(len(self.clients), self.z_dir + self.partition_data_file)

        else:
            print("No partition_data_file!")
            return 
        

        print("Distributing data...")

        for client_id, client_data in tqdm(self.train_data.items()):
            self.clients[client_id].data = client_data
            self.data_size.append(len(client_data))
          
        lr = self.learning_rate

        for epoch in range(self.global_rounds):
            print(f"Epoch {epoch+1}/{self.global_rounds}")
            for i, client in tqdm(enumerate(self.clients)):
                client.train(device, lr)
            #lr = lr * self.lr_decay


        self.create_distance_file()
        self.create_client_wPCA2D_file()

        s = sum(self.data_size)
        print("data_size: ", self.data_size)
        train_acc_avg, train_loss_avg = 0,0

        print("Training statistic: ")
        for i in range(len(self.clients)):
            train_loss, train_acc = self.clients[i].train_stats(device)
            train_loss_avg += train_loss * len(self.clients[i].data)/s
            train_acc_avg += train_acc * len(self.clients[i].data)/s
            
            print("Client ", i, ": Accuracy ","{0:.10}%".format(train_acc) , " Loss: ", f'{train_loss:.10}')

        print("Accuracy_avg: ", train_acc_avg, "Loss_avg: ", train_loss_avg)
        return

    def generate_clients(self, n_clients):
        print("Generate client nodes!!!")
        clients = []
        for i in range(n_clients):
            client = ClientNode(self.learning_rate)
            model = copy.deepcopy(self.model)
            client.model["model"] = model
            clients.append(client)

        return clients

    def create_client_wPCA2D_file(self):
        #create client weights by PCA in 2D file
        W = []
        for i in tqdm(range(len(self.clients))):

            w = torch.tensor([]).to(device)
            model = self.clients[i].model["model"]
            with torch.no_grad():
                for param in model.parameters():
                    w = torch.cat((w, torch.flatten(param.data)), 0)
            
            #print("w[", i, "]", w[0:50])
            W.append(w.to('cpu').numpy())
        W = np.asarray(W)
        #print("W shape: ", W.shape)
        #print("W before StandardScaler: ", W[0:10][0:5], "\n\n") 
        
        #print("W before StandardScaler: ", W[10:20][0:5]) 
        
        W = StandardScaler().fit_transform(W)
        #print("W shape: ", W.shape)
        #print("W after standanScalser ", W[0:11][0:5])
        
        pca = PCA(n_components=2)

        principalComponents = pca.fit_transform(W)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

        print("creating weights in 2D...")
        
        if not os.path.exists(self.z_dir):
            os.makedirs(self.z_dir)
        
        wPCA_fname = "wPCA_" + self.model_name + "_G" + str(self.global_rounds) + "_" + self.partition_data_file

        principalDf.to_csv(self.z_dir + wPCA_fname)
        print("saved in file: ", self.z_dir + wPCA_fname)

        
    def create_distance_file(self):
        
        print("creating distance file...")
        
        if not os.path.exists(self.z_dir):
            os.makedirs(self.z_dir)
        
        if self.p == 2:  
            d_fname = "d_euclidean_" +self.model_name + "_G" + str(self.global_rounds) + "_" + self.partition_data_file 
        else:
            d_fname = "d_" + self.distance_metric + "_p" + str(self.p) + "_" +self.model_name + "_G" + str(self.global_rounds) + "_" + self.partition_data_file
         
        d = self.client_distance_matrix()
        df = pd.DataFrame(d)
        df.to_csv(self.z_dir + d_fname, header = False, index = False)
        
        print("saved in file: ", self.z_dir + d_fname)
        
        return

    def client_distance_matrix(self):
        dist_matrix = np.zeros((len(self.clients), len(self.clients)))

        for i in tqdm(range(len(self.clients))):
            
            model_A = self.clients[i].model["model"]
            
            for j in range(i+1, len(self.clients)):
                              
                model_B = self.clients[j].model["model"]
                
                if self.distance_metric == 'euclidean':
                    dist_matrix[i][j] = self.weight_euclidean_difference(model_A, model_B) #self.weight_similarity_by_RBF_kernel(model_A, model_B)
                else:
                    dist_matrix[i][j] = self.weight_minkowski_difference(model_A, model_B)
                dist_matrix[j][i] = dist_matrix[i][j]
        #print(dist_matrix)
        #exit()
        return dist_matrix

    def weight_euclidean_difference(self, model_A, model_B):
        dot_product = 0
        with torch.no_grad():
            for param_A, param_B in zip(model_A.parameters(), model_B.parameters()):
                dot_product += torch.dot(torch.flatten(param_A.data) - torch.flatten(param_B.data), torch.flatten(param_A.data) - torch.flatten(param_B.data))
                
        return math.sqrt(dot_product)

    def weight_minkowski_difference(self, model_A, model_B):
        if self.p==0:
            return 0
        s = 0
        w = torch.tensor([]).to(device)
        with torch.no_grad():
            for param_A, param_B in zip(model_A.parameters(), model_B.parameters()):
                
                w = torch.cat((w, torch.flatten(param_A.data) - torch.flatten(param_B.data)), 0)
                
        d = sum(np.abs(w.to('cpu').numpy()) ** self.p)
            
                
        
        return d ** (1.0/self.p)
