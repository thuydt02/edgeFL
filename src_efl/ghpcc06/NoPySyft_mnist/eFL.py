import random
import pandas as pd
#from sklearn.metrics.pairwise import cosine_similarity
import math

import os
import torch
import copy
import numpy as np
from tqdm import tqdm


from torch import nn

#from model import SimpleCNN
#from model import SimpleCNNMNIST
#from model import lCNNMNIST
#from model import EfficientNetB0
#from model import CNN2
#from model import CNN3
#from model import MLP3


#from mnist_model import CNN1 #CNNModel
#from mnist_model import CNNModel
#from mnist_model import CNNModel2
from mnist_model import MLP2

from edge_server_node import EdgeServerNode
from client_node import ClientNode
from data_loader import Data_Loader
#from data_loader_efficientnet_cifar100 import Data_Loader_EfficientNet_Cifar100 

#from torchsummary import summary

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
#device = "cpu"


class eFL:
    def __init__(self, no_edge_servers, mode , no_clients, num_epochs, batch_size, learning_rate, lr_decay, edge_update, global_update, 
        w_dir, z_dir, acc_dir, z_file,
        dataset, pre_trained_w_file = None, partition_data_file = None,
        q_file = None):

        #self.model =  CNNModel().to(device) #MNIST FEMNIST
        #summary(self.model, (1, 28, 28))
        #self.model_name = "CNN"

        #mnist
        self.model =  MLP2().to(device) #MNIST().to(device) #CNNModel().to(device) MNIST FEMNIST
#        summary(self.model, (1, 28, 28))
        self.model_name = "MLP2"

        
        #self.model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(device) #cifar10, cifar100
        #summary(self.model, input_size=(3, 32, 32)) #channel, img_size, img_size
        #self.model_name = "sCNN"

        #self.model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(device) #cifar100
        #summary(self.model, input_size=(1, 28, 28)) #channel, img_size, img_size
        #self.model_name = "sCNNMNIST"


        #self.model = CNN2(hidden_dims=[256, 128], output_dim=10).to(device) #cifar100, cifar10
        #summary(self.model, input_size=(3, 32, 32)) #channel, img_size, img_size
        #self.model_name = "CNN2_dr10"

        #self.model = CNN3().to(device) #cifar10
        #summary(self.model, input_size=(3, 32, 32)) #channel, img_size, img_size
        #self.model_name = "CNN3dr10"

        
        #self.model = ModerateCNN(output_dim = 10).to(device) #cifar10
        #summary(self.model, input_size=(3, 32, 32)) #channel, img_size, img_size
        #self.model_name = "mCNN"

        #self.model =  MLP3().to(device) #MNIST().to(device) #CNNModel().to(device) MNIST FEMNIST
        #summary(self.model, (3, 32, 32))
        #self.model_name = "MLP3"
        #self.dataset = dataset # mnist, femnist 
        self.partition_data_file = partition_data_file
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        
        self.edge_servers = self.generate_edge_servers(no_edge_servers)
        self.clients = self.generate_clients(no_clients)
        self.mode = mode

        self.data_loader = Data_Loader(dataset, batch_size)#Data_Loader_EfficientNet_Cifar100(batch_size)
        
        self.train_data = None
        self.test_data = None
        self.data_size = [] # for aggregate model
        self.assignment = None

        self.edge_update = edge_update
        self.global_update = global_update


        self.w_dir = w_dir
        self.z_dir = z_dir
        self.acc_dir = acc_dir 

        self.z_file = z_file
        if "rnd" in self.z_file:
            if q_file != None:            
                self.w_file = "eFL_q" + self.model_name + "lr" + str(self.learning_rate) + "_dc" + str(self.lr_decay)+ "_B" + str(batch_size) + "_L" + str(edge_update) + "_E" + str(global_update) + "_G" + str(num_epochs) + "_"+ self.z_file + "_" + partition_data_file + q_file+ ".weight.pth"
            else:
                self.w_file = "eFL_" + self.model_name  + "lr" + str(self.learning_rate) + "_dc" + str(self.lr_decay)+  "_B" + str(batch_size) + "_L" + str(edge_update) + "_E" + str(global_update) + "_G" + str(num_epochs) + "_"+ self.z_file + "_" + partition_data_file + ".weight.pth"
        else:
            if q_file != None:
                self.w_file = "eFL_q" + self.model_name + "lr" + str(self.learning_rate) + "_dc" + str(self.lr_decay) + "_B" + str(batch_size) + "_L" + str(edge_update) + "_E" + str(global_update) + "_G" + str(num_epochs) + "_"+ self.z_file + ".weight.pth"
            else:
                self.w_file = "eFL_" + self.model_name + "lr" + str(self.learning_rate) + "_dc" + str(self.lr_decay) + "_B" + str(batch_size) + "_L" + str(edge_update) + "_E" + str(global_update) + "_G" + str(num_epochs) + "_"+ self.z_file + ".weight.pth"
            
        self.acc_file = self.w_file + ".acc.csv"
        
        if q_file != None:
            self.q = np.load(z_dir + q_file)
        else:
            self.q = np.ones(no_clients)

        self.p = []  # will be computed in load_assignment()
        self.list_selected_client = np.zeros(no_clients, dtype = bool)

        self.z = None   # the array of assignment: z[i] = m means client i is assigned to sever m
        self.LAMBDA = 0 # the trade_off coefficient between communication cost and accuracy of model.
        #self.c = self.calculate_distance_matrix() # the communication cost matrix: c[i][m] = the communication cost from client i to server m
        self.d = None #calculate_weight_difference_matrix() #the weight distance matrix d[i][j] = the difference betweet model i and model j. Model i belongs to client i
        self.D = None #the array of current weight-distance cost of each edge server. D[0] = 100 means edge server 0 has weight distance cost = 100
        

        print ("Configuration: ")
        print ("num_clients, num_edges: ", no_clients, ", ", no_edge_servers)
        print("L, E, G: ", edge_update, ", ", global_update, ", ", num_epochs)
        print("B, lr, lr_decay: ", batch_size, ", ", learning_rate, ", ", lr_decay)
        print("dataset, model: ", dataset, ", ", ", ", self.model_name)
        print("partition_data_file: ", partition_data_file)

        if (pre_trained_w_file != None):
            print("Loading weight from " + pre_trained_w_file)
            self.model.load_state_dict(torch.load(self.w_dir + pre_trained_w_file)) 

         

#----------------------
#Thuy: begin{new code area}
#----------------------

    
    def average_models(self, models, coefficients):
        averaged_model = copy.deepcopy(self.model)
        p = np.asarray(coefficients)/sum(coefficients)

        with torch.no_grad():
            averaged_values = {}
            for name, param in averaged_model.named_parameters():
                averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))
            i = 0
            for model in models:
                for name, param in model.named_parameters():
                    averaged_values[name] += p[i] * param.data
                i += 1

            for name, param in averaged_model.named_parameters():
                param.data = averaged_values[name]

        return averaged_model

    def copy_model(self, source_model):
        model_copy = type(source_model)()
        model_copy.load_state_dict(source_model.state_dict())

        return model_copy

    def send_model_to_edge_servers(self):
        for edge_server in self.edge_servers:
            edge_server.model = copy.deepcopy(self.model)


    def send_model_to_clients(self):
        for edge_server in self.edge_servers:
            for client_id in edge_server.connected_clients:
                model = copy.deepcopy(edge_server.model) #self.copy_model(edge_server.model).to(device)
            
                client = self.clients[client_id]
                #client.clear_model()
                client.model["model"] = model


    def generate_edge_servers(self, no_edge_servers):
        print("Generate edge server nodes")
        edge_servers = []
        for i in range(no_edge_servers):
            edge_server = EdgeServerNode(self.model)
            edge_servers.append(edge_server)

        return edge_servers


    def generate_clients(self, no_clients):
        print("Generate client nodes")
        clients = []
        for i in range(no_clients):
            client = ClientNode(self.learning_rate)
            clients.append(client)

        return clients

    def calculate_distance_matrix(self):
        edge_server_locations = [edge_server.location for edge_server in self.edge_servers]

        distance_matrix = []
        
        for client in self.clients:
            distances = [np.linalg.norm(client.location - edge_server_location) for edge_server_location in edge_server_locations]
            distance_matrix.append(distances)
        
        return np.array(distance_matrix)

#---------------------------------------------------
#Thuy begin{new code area}
#---------------------------------------------------
	
    

    def create_client_data_size_in_batch_file(self):

        print("creating client data size in batch file...")
        fname = "client_data_size_" + self.partition_data_file

        with open(self.z_dir + fname, 'w') as the_file:
            the_file.write(str(len(self.clients[0].data)) + "\n")
        
        for i in range (1,len(self.clients)):
            
            with open(self.z_dir + fname, 'a') as the_file:
                the_file.write(str(len(self.clients[i].data)) + "\n")
        
        the_file.close()
        print("saved client data size in file: " + self.z_dir + fname)

        return

    
    def load_assignment(self):
        f = open(self.z_dir + self.z_file, 'r')
        self.z = np.asarray([int(line.strip()) for line in f.readlines()])
        f.close()
        
        for es in self.edge_servers:
            es.connected_clients = []
            es.data_size = []
        
        for i in range (len(self.clients)):
            self.edge_servers[self.z[i]].connected_clients.append(i)
            self.edge_servers[self.z[i]].data_size.append(len(self.clients[i].data))

        for es in self.edge_servers:
            self.data_size.append(sum(es.data_size))
            
        s = sum(self.data_size)
        self.p = [len(cl.data)/s for cl in self.clients]

        print("cloud server: data_size = ", self.data_size)
        

	
    

    def train(self):

        

        print("Start training...")

        if  not os.path.exists(self.acc_dir):
            os.makedirs(self.acc_dir)

        if  not os.path.exists(self.acc_dir + self.mode):
            os.makedirs(self.acc_dir + self.mode)

        if  not os.path.exists(self.acc_dir + self.mode + "/L" + str(self.edge_update) + "E" + str(self.global_update)):
            os.makedirs(self.acc_dir + self.mode + "/L" + str(self.edge_update) + "E" + str(self.global_update))

        acc_path = self.acc_dir + self.mode + "/L" + str(self.edge_update) + "E" + str(self.global_update) + "/"
            
        
        self.send_model_to_edge_servers()
        self.send_model_to_clients()
        

        save_at = []
        n_save_at = int(self.num_epochs / 500)

        for i in range(n_save_at):
            save_at.append( (i + 1) * 500 )
            
        with open(acc_path + self.acc_file, 'w') as the_file:
            the_file.write("global_round,train_loss,train_acc,test_loss,test_acc\n")
        
        lr = self.learning_rate

        self.list_selected_client = self.select_clients()
        print("p: sum = ", sum(self.p))
        print(self.p)

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
    
            for i, client in tqdm(enumerate(self.clients)):
                if self.list_selected_client[i]:
                    client.train(device, lr)            
            #lr = lr * self.lr_decay # for mnist + MPL2 with init self.learning_rate = 0.01, decay = 0.995

            #lr = self.learning_rate * (1 / (1 + self.lr_decay * (epoch % self.edge_update))) #cifar10 (1)
            #if lr < 0.001:
            #    lr = 0.001 #cifar10

            # Average models at edge servers
            if (epoch+1) % self.edge_update == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}")
                print("---- [UPDATE MODEL] Send local models to edge servers ----")
                for edge_server in self.edge_servers:
                    #models = [self.clients[client_id].model["model"] for client_id in edge_server.connected_clients]
                    
                    if len(edge_server.connected_clients) <= 0:
                        continue

                    models = []
                    edge_server.data_size = []
                    for cl in edge_server.connected_clients:
                        if self.list_selected_client[cl]:
                            models.append(self.clients[cl].model["model"])
                            edge_server.data_size.append(len(self.clients[cl].data))
                    
                    if len(models) > 0:
                        edge_server.model = self.average_models(models, edge_server.data_size)

                lr = lr * self.lr_decay #for mnist + MPL2
                #if epoch + 1 < 120:
                #lr = self.learning_rate * (1.0 / (1.0 + self.lr_decay * ((epoch + 1) / self.edge_update)))
                #else:
                #    lr = 0.0015
                
                if (epoch+1) % (self.global_update * self.edge_update) != 0:
                    print("----[DELIVER MODEL] send edge models -> clients")
                    self.send_model_to_clients()
                    self.list_selected_client = self.select_clients()
                
            # Average models at cloud servers
            
            if (epoch+1) % (self.global_update * self.edge_update) == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}")
                print("---- [UPDATE MODEL] Send edge models to cloud server ----")
                
                #models = [edge_server.model for edge_server in self.edge_servers if len(edge_server.connected_clients) > 0]
                
                models = []
                self.data_size = []
                for es in self.edge_servers:
                    if len(es.connected_clients) <= 0:
                        continue

                    if sum(self.list_selected_client[edge_server.connected_clients] > 0):
                        models.append(edge_server.model)
                        self.data_size.append(sum(np.asarray(edge_server.data_size)))
                if len(models) > 0:
                    self.model = self.average_models(models, self.data_size)
                    print("----[DELIVER MODEL] send global model -> edges")
                    self.send_model_to_edge_servers()
                    print("----[DELIVER MODEL] send edge models -> clients")
                    self.send_model_to_clients()
                    self.list_selected_client = self.select_clients()

                train_loss, train_acc = self.train_stats()
                
                print("Training statistic: ")
                print("Accuracy ","{0:.10}%".format(train_acc) , " Loss: ", f'{train_loss:.10}')
                
                # Validate new model
                test_loss, test_acc = self.validate(load_weight=False)
                print ("lr: ", lr)
                
                with open(acc_path + self.acc_file, 'a') as the_file:
                    the_file.write(str(epoch+1) + "," + str(train_loss) + "," + str(train_acc) + "," + str(test_loss) + "," + str(test_acc)+ "\n")
                    #the_file.write(str(epoch+1) + ",0" + ",0" + "," + str(test_loss) + "," + str(test_acc)+ "\n")
            
#            if epoch in save_at:
#                fmodel = str(epoch) + "_" + self.w_file
#                self.save_model(fmodel)

        print("Finish training!")
        the_file.close()
        
                
        
    def validate(self, load_weight=False):

        print("Validation statistic...")

        if load_weight == True:
            self.model.load_state_dict(torch.load(self.w_dir + self.w_file))

        self.model.eval()
        corrects = 0
        loss = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_data):
                images, labels = images.to(device), labels.to(device)
                output = self.model(images)
                pred = output.argmax(dim=1)
                corrects += pred.eq(labels.view_as(pred)).sum().item()

                loss += nn.CrossEntropyLoss()(output, labels).item()


        total_test = len(self.test_data)*self.batch_size
        accuracy = 100*corrects/total_test
        loss = loss/len(self.test_data)

        print("Number of corrects: {}/{}".format(corrects, len(self.test_data)*self.batch_size))
        print("Accuracy, {}%".format(accuracy), " Loss: ", f'{loss:.3}')
        print("-------------------------------------------")

        return loss, accuracy 

    def train_stats(self):
        train_acc = 0.0
        train_loss = 0.0
                
        for i, cl in enumerate(self.clients):
            #if self.list_selected_client[i]:
            loss, acc = cl.train_stats(device)
            train_acc += self.p[i] * acc #* self.q[i]
            train_loss += self.p[i] * loss # * (1 / self.q[i])
            

        return train_loss, train_acc    
    
    def save_model(self, fmodel):
        print("Saving model...")
        if not os.path.exists(self.w_dir):
            os.makedirs(self.w_dir)
        torch.save(self.model.state_dict(), self.w_dir + fmodel)
        print("Model saved!")
        
        
    
    
    def run_train(self):
        # load and distribute data to clients
        print("Creating data distribution for clients...")
        if (self.partition_data_file != None):
                print("Partition_data_file: ", self.partition_data_file)
                self.train_data = self.data_loader.prepare_in_batch_given_partition(len(self.clients), self.z_dir +  self.partition_data_file)
        else:
            print("No partition_data_file")
            return
    
        self.test_data = self.data_loader.test_data

        print("Distributing data...")
        for client_id, client_data in tqdm(self.train_data.items()):
            self.clients[client_id].data = client_data
            #print("client, num_batch ", client_id, ", ", len(client_data))
        
        print("Loading z assignment from ", self.z_dir + self.z_file)
        self.load_assignment()

        
        print("Computing label distribution across clients: ")
        for i, cl in enumerate(self.clients): 
            list_labels = cl.get_labels(device)

            print("client  ", i, " num_labels: ", len(list_labels), " ",  list_labels)

        
        print("Computing label distribution edge servers: ")
        for m, es in enumerate(self.edge_servers):
            list_labels = []
            for cl_id in es.connected_clients: 
                list_labels += self.clients[cl_id].set_labels
            es.list_labels = list(set(list_labels))
            print("edge server ", m, " num_labels: ", len(es.list_labels), " ",  es.list_labels)

        

        self.train()

    def select_clients(self):
        p_select_client = np.random.random_sample((len(self.clients), ))
        a = (p_select_client <= self.q)
        print("num_selected_clients: ", sum(a))
        return a
























#------------------------------------------------------------------
#------------------------------------------------------------------

#helping functions, not used now

    def save_assignment(self, rd = False):
        print("Saving assignment...")
        if not os.path.exists(self.w_dir):
            os.makedirs(self.w_dir)
        #fname =
        #pd.to_csv()
        print("Model saved!")
    
    def init_assignment(self):
        #reandomly initialize an assignment client i -> edge server m
        #n cells
        #m servers
        #each cell is assigned to one server. each edge server has at least 1 client
        
        n = len(self.clients)
        m = len(self.edge_servers)
        
        self.z = np.empty(n, dtype=int)
        mask = np.zeros(n, dtype=bool)
        tmp = random.sample(set(np.arange(n)), m)
        for s in range (m):
            self.z[tmp[s]] = s
        remaining = set(np.arange(n)) - set(tmp)
        for i in remaining:
            s = random.randint(0,m-1)
            self.z[i] = s
        
        
        for es in self.edge_servers:
            es.connected_clients = []
            es.data_size = []
        
        for i in range (len(self.clients)):
            self.edge_servers[self.z[i]].connected_clients.append(i)
            self.edge_servers[self.z[i]].data_size.append(len(self.clients[i].data))

        for es in self.edge_servers:
            self.data_size.append(sum(es.data_size))

        s = sum(self.data_size)
        self.p = [len(cl.data)/s for cl in self.clients]

        print("cloud server: data_size = ", self.data_size)

    def shifting(self):
        n = len(self.clients)
        m = len(self.edge_servers)
        print('shifting: assignment ', self.z)
        
        for i in range(n):
            s_star = self.z[i]
            if len(self.edge_servers[s_star]) <= 1:
                continue
            gain = 0
            best_s = -1
            #looking for the best cluster for i
            
            for s in range(m):
                if s == s_star:
                    continue
                #compute the different of distance costs if we move i from s_star to s
                
                dcost1 = np.sum([self.d[i][j] **2 for j in self.edge_servers[s_star].connected_clients])
                
                D_sstar_new = (self.D[s_star] * len(self.edge_servers[s_star]) - dcost1) / (len(self.edge_servers[s_star]) - 1)
                
                dcost2 = np.sum([self.d[i][j] **2 for j in self.edge_servers[s].connected_clients])
                
                D_snew = (D[s] * len(self.edge_servers[s]) + dcost2)/(len(self.edge_servers[s]) + 1)
                
                
                ccost_gain = self.c[i][s] - self.c[i][s_star]
                dcost_gain = D_snew + D_sstar_new - self.D[s] - self.D[s_star]
                
                new_gain = self.LAMBDA * ccost_gain + (1-self.LAMBDA) * dcost_gain
                
                print ('i, s, new_gain = ', i, ', ',s, ', ', new_gain)
                
                if (new_gain < 0):
                    gain = new_gain
                    best_s = s
                print('best_s: ', best_s)
            
            if best_s != -1:
                self.z[i] = best_s
                self.edge_servers[s_star].remove(i)
                self.edge_servers[best_s].append(i)
                self.D[s_star] = self.get_weight_cost_given_edge_server(self.edge_servers[s_star])
                self.D[best_s] = self.get_weight_cost_given_edge_server(self.edge_servers[best_s])
                
                print('moved: ', i, ' from ', s_star, ' to ', best_s)
        return

    def BR(self):
        #communication cost matrix: c[i,s] = communication cost to transfer the data from cell i to server s
        #lb: coeffiectient for trade-off between communication cost and the weight distances
        #d: a matrix, d[i][j] = the weight distance between cell i and cell j
        #return the assignment z
        n = len(self.clients)
        m = len(self.edge_servers)
        
        #randomly initialize a assignment
        self.init_assignment()

        #for convenience
        #cluster: a list of list of clients. for example cluster[1] = [1, 2,, 3] means : cluster number 1 has 3 client assigned {1, 2,3}
        
        cluster = np.empty(m, dtype=list)
        for s in range(m):
            cluster[s] = []
        for i in range (n):
            cluster[z[i]].append(i)

        #D an array represents the d_cost of each cluster
        D = np.zeros(m, dtype=float)
        for s in range(m):
            D[s] = get_dcost_1_cluster(cluster[s], d)
        #shifting at first
        #swaping secondly
        
        total_cost = communication_cost(z, c, ld) + weight_cost(z,d, ld)
        
        for k in range(3):
            shifting(cluster, D, z, c, d, ld)
            #swapping()
            
        return z

    def get_weight_cost_given_edge_server(self, es):
    # return the weight_cost of a edge server es
        if self.d == None:
            self.d = self.calculate_weight_difference_matrix()
        D = 0
        for i in range(len(es.connected_clients)):
            for j in range(i + 1, len(es.connected_clients), 1):
                D += d[es.connected_clients[i]][es.connected_clients[j]] **  2
        return D / len(es.connected_clients)
    
    def communication_cost(self):
        
        #to compute the communication cost of an assignment
        if self.c == None:
            self.c = self.calculate_distance_matrix()
        
        n = len(self.z) # #cells
        m = len(self.edge_servers) # #servers
        retval = 0
        
        for i in range(n):
            retval += self.c[i][self.z[i]]
        return retval #self.LAMBDA * retval

    def weight_cost(self):
        #compute the weight distances of all edge server
        # the objective of kmeans
        
        if self.d == None:
            self.d = self.calculate_weight_difference_matrix()
        
        n = len(self.z) # #cells
        m = len(self.edge_servers) # #servers
        if self.D == None:
            self.D = np.empty(m, dtype=float)
            for s in range(m):
                self.D[s] += self.get_weight_cost_given_edge_server(self.edge_servers[s])
        return np.sum(self.D) #(1-self.LAMBDA)* np.sum(self.D)

    
