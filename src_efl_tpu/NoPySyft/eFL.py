import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import math

import os
import torch
import copy
import numpy as np
from tqdm import tqdm


from torch import nn

from mnist_model import CNNModel
from edge_server_node import EdgeServerNode
from client_node import ClientNode
from data_loader import MNISTDataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

#gg colab
#acc_DIR = "/content/drive/MyDrive/NA-Multiple-Edge-Servers-Federated-Learning/acc/"
#z_DIR = "/content/drive/MyDrive/NA-Multiple-Edge-Servers-Federated-Learning/z_ass/"
#local
#acc_DIR = "../../acc/"
#z_DIR = "/../..z_ass/"

#z_FILENAME = "graph_weight_euclidean_diff.txt.part.10"

#z_FILENAME = "g_d_euclidean_L50_non_iid_zipfz0.7.part.8" #z_metis
#z_FILENAME = "2z_rnd.part8" 
#w_FILENAME = z_FILENAME + ".weight.pth"
#acc_FILENAME = z_FILENAME + ".acc.csv"

class eFL:
    def __init__(self, no_edge_servers, no_clients, num_epochs, batch_size, learning_rate, edge_update, global_update, w_dir, z_dir, acc_dir, z_file, mode, zipfz):
        self.model = CNNModel().to(device)

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.edge_servers = self.generate_edge_servers(no_edge_servers)
        self.clients = self.generate_clients(no_clients)

        self.data_loader = MNISTDataLoader(batch_size)
        self.zipfz = zipfz
        self.train_data = None
        self.data_size = []
        self.assignment = None

        self.edge_update = edge_update
        self.global_update = global_update


        self.w_dir = w_dir
        self.z_dir = z_dir
        self.acc_dir = acc_dir 
        
        self.mode = mode

        self.z_file = z_FILE            
        self.w_file = "eFL_" + mode + "_B" + str(batch_size) + "_L" + str(edge_update) + "_E" + str(global_update) + "_G" + str(num_epochs) + "_MNIST_zipfz" + str(zipfz) + "_"+ self.z_file + ".weight.pth"
        self.acc_file = self.w_file + ".acc.csv"

#----------------------
#Thuy: begin{new code area}
#----------------------
        self.z = None   # the array of assignment: z[i] = m means client i is assigned to sever m
        self.LAMBDA = 0 # the trade_off coefficient between communication cost and accuracy of model.
        self.c = self.calculate_distance_matrix() # the communication cost matrix: c[i][m] = the communication cost from client i to server m
        self.d = None #calculate_weight_difference_matrix() #the weight distance matrix d[i][j] = the difference betweet model i and model j. Model i belongs to client i
        self.D = None #the array of current weight-distance cost of each edge server. D[0] = 100 means edge server 0 has weight distance cost = 100
        self.acc_logs = []
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
        print("Send model to edge servers")
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
	
    def weight_cosine_difference(self, model_A, model_B):
        #cosine_similarity = 1 - spatial.distance.cosine(model_A.parameters, model_B.parameters)
        dot_product = 0
        magnitude_A = 0
        magnitude_B = 0
        
        model_A_vector, model_B_vector = [], []
        with torch.no_grad():
            for param_A, param_B in zip(model_A.parameters(), model_B.parameters()):
                dot_product += torch.dot(torch.flatten(param_A.data), torch.flatten(param_B.data))
                magnitude_A += torch.dot(torch.flatten(param_A.data), torch.flatten(param_A.data))
                magnitude_B += torch.dot(torch.flatten(param_B.data), torch.flatten(param_B.data))
                
        return int(1000 * dot_product / (math.sqrt(magnitude_A) * math.sqrt(magnitude_B)))
            
    def weight_euclidean_difference(self, model_A, model_B):
        dot_product = 0
        with torch.no_grad():
            for param_A, param_B in zip(model_A.parameters(), model_B.parameters()):
                dot_product += torch.dot(torch.flatten(param_A.data) - torch.flatten(param_B.data), torch.flatten(param_A.data) - torch.flatten(param_B.data))
                #magnitude_A += torch.dot(torch.flatten(param_A.data), torch.flatten(param_A.data))
                #magnitude_B += torch.dot(torch.flatten(param_B.data), torch.flatten(param_B.data))
                
        return 1 + int(1000*math.sqrt(dot_product))
    
    def weight_difference_matrix(self):
        difference_matrix = np.zeros((len(self.clients), len(self.clients)))

        for i in tqdm(range(len(self.clients))):
            #client_A = self.clients[i]
            model_A = self.clients[i].model["model"] #client_A.model["model"]
            
            for j in range(i+1, len(self.clients)):
                
                #client_B = self.clients[j]
                if j == i:
                    continue
                model_B = self.clients[j].model["model"]#client_B.model["model"]

                #difference = self.weight_cosine_difference(model_A, model_B) # for cosine dissim
                difference = self.weight_euclidean_difference(model_A, model_B) # for eucl dissim
                
                difference_matrix[i][j] = difference
                difference_matrix[j][i] = difference

        return difference_matrix
        
    def create_graph(self, no_local_epochs):
        print("creating z graph...")
        if not os.path.exists(self.w_dir):
            os.makedirs(self.w_dir)
        if self.zipfz != None:
            z_matrix_fname = "d_euclidean_L" + str(no_local_epochs) + "_"+ self.mode +"_zipfz" + str(self.zipfz)
        else:    
            z_matrix_fname = "d_euclidean_L" + str(no_local_epochs)
        fname = "g_" + z_matrix_fname #50 local eporch,
        
        d = self.weight_difference_matrix()
        #save d matrix
        df = pd.DataFrame(d)
        df.to_csv(self.z_dir + z_matrix_fname)
        
        
        num_edges = len(self.clients) * (len(self.clients) -1) / 2
        header = str(len(self.clients)) + " " + str(int(num_edges)) + " 011\n"
        with open(self.z_dir + fname, 'w') as the_file:
            the_file.write(header)
        for i in range (len(self.clients)):
            i_prime = i + 1
            a_line = str(len(self.clients[i].data))
            for j in range (len(self.clients)):
                j_prime = j + 1
                if j_prime == i_prime:
                    continue
                a_line = a_line + " " + str(j_prime) + " " + str(int(d[i][j]))
            a_line = a_line + "\n"
            with open(self.z_dir + fname, 'a') as the_file:
                the_file.write(a_line)
        the_file.close()
        print("saved z graph!")
    
    
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
            
            #print ("es: clients ", es.connected_clients)
            #es_data = [len(self.clients[cl].data) for cl in es.connected_clients]
            #print("es.data_size: ", es.data_size)
            #print("es_data: ", es_data)

        print("cloud server: data_size = ", self.data_size)
        #exit()
        #for i in range(len(self.edge_servers)):
        #    print("edge server ",i , ": data_size: ", self.edge_servers[i].data_size)
        #    print("edge_server ", i, "connected_clients: data_size [")
        #    for cl_id in self.edge_servers[i].connected_clients:
        #        print(len(self.clients[cl_id].data), ", ")
        


	
    def pre_train_for_z_graph(self, no_local_epochs):
        
        print("Creating data distribution for clients...")
        if (self.mode == "non_iid"):
            if (self.zipfz == None):
                self.train_data = self.data_loader.prepare_non_iid_data_option1(len(self.clients))
            else:
                self.train_data = self.data_loader.prepare_non_iid_data_option_zipf(len(self.clients), self.zipfz)
        else:
            self.train_data = self.data_loader.prepare_iid_data(len(self.clients))

        print("Distributing data...")
        for client_id, client_data in tqdm(self.train_data.items()):
            self.clients[client_id].data = client_data

        # Send model to edge server
        #self.send_model_to_edge_servers()
        #is_updated = True
        print("---- Assignment Phase Model Training ----")

        # Send models from edge to nearest workers
        #shortest_distance_assignment = self.shortest_distance_clients_servers_assign()
        self.init_assignment()
        #print("edge_servers.connected_client: ", self.edge_servers[0].connected_clients)
        print("-- Send edge server models to workers --")
        self.send_model_to_edge_servers()
        self.send_model_to_clients()

        # Train the local models for a few epochs
        #no_local_epochs = 1
        
        for epoch in range(no_local_epochs):
            print(f"Epoch {epoch+1}/{no_local_epochs}")
            # Train each worker with its own local data
            for i, client in tqdm(enumerate(self.clients)):
                client.train(device)

        self.create_graph(no_local_epochs)
        return
            
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


    #helping functions

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
        for i in range (n):
            self.edge_servers[self.z[i]].connected_clients.append(i)
        
        
        df = pd.DataFrame(self.z.tolist())
        df.to_csv(self.z_dir + z_FILENAME)
        return self.z

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
#---------------------------------------------------
#Thuy end{new code area}
#---------------------------------------------------


    def train(self):

        
        print("Start training...")
        
        # Send model to edge server
        self.send_model_to_edge_servers()
        
        is_updated = True    
        best_acc = 0


        with open(self.acc_dir + self.acc_file, 'w') as the_file:
            the_file.write("global_round,train_loss,train_acc,test_loss,test_acc\n")
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            # Send the edge servers' models to all the workers
            if is_updated:
                print("---- [DELIVER MODEL] Send edge server models to clients ----")
                self.send_model_to_clients()
                is_updated = False

            # Train each worker
            for i, client in tqdm(enumerate(self.clients)):
                client.train(device)

        
            # Average models at edge servers
            if (epoch+1) % self.edge_update == 0:
                print("---- [UPDATE MODEL] Send local models to edge servers ----")
                is_updated = True
                for edge_server in self.edge_servers:
                    models = [self.clients[client_id].model["model"] for client_id in edge_server.connected_clients]
                    edge_server.model = self.average_models(models, edge_server.data_size)

                #for client in self.clients:
                #    client.clear_model()

            # Average models at cloud servers
            if (epoch+1) % self.global_update == 0:
                print("---- [UPDATE MODEL] Send edge servers to cloud server ----")
                models = [edge_server.model for edge_server in self.edge_servers if len(edge_server.connected_clients) > 0]
                self.model = self.average_models(models, self.data_size)

                train_loss, train_acc = self.train_stats()
            
                print("Training statistic: ")
                print("Accuracy ","{0:.10}%".format(train_acc) , " Loss: ", f'{train_loss:.10}')
            
            # Validate new model
                test_loss, test_acc = self.validate(load_weight=False)

                with open(self.acc_dir + self.acc_file, 'a') as the_file:
                    the_file.write(str(epoch+1) + "," + str(train_loss) + "," + str(train_acc) + "," + str(test_loss) + "," + str(test_acc)+ "\n")
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.save_model()
                
                # Send the global model to edge servers
                print("---- [DELIVER MODEL] Send global model to edge servers ----")
                self.send_model_to_edge_servers()
                is_updated = True
            
            #if ((epoch + 1) % self.edge_update == 0) or ((epoch + 1) % self.global_update == 0):   
            #    for client in self.clients:
            #        client.clear_model()
                
                
        #self.acc_logs = accuracy_logs
        print("Finish training!")
        print ("saving acc_log")
        the_file.close()
        #self.save_acc_loss_logs(acc_DIR + acc_FILENAME)
        #self.save_assignment(True)
        #   self.save_acc_logs()


    def validate(self, load_weight=False):

        print("Validation statistic...")

        if load_weight == True:
            self.model.load_state_dict(torch.load(self.w_dir + self.w_file))

        self.model.eval()
        corrects = 0
        loss = 0

        test_data = self.data_loader.test_data
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_data):
                images, labels = images.to(device), labels.to(device)
                output = self.model(images)
                pred = output.argmax(dim=1)
                corrects += pred.eq(labels.view_as(pred)).sum().item()

                loss += nn.CrossEntropyLoss()(output, labels).item()


        total_test = len(test_data)*self.batch_size
        accuracy = 100*corrects/total_test
        loss = loss/len(test_data)

        print("Number of corrects: {}/{}".format(corrects, len(test_data)*self.batch_size))
        print("Accuracy, {}%".format(accuracy), " Loss: ", f'{loss:.3}')
        print("-------------------------------------------")

        return loss, accuracy 

    def train_stats(self):
        train_acc = 0
        train_loss = 0
        s = sum(self.data_size)
        
        p = [len(cl.data)/s for cl in self.clients]
        i = 0
        for cl in self.clients:
            loss, acc = cl.train_stats(device)
            train_acc += p[i] * acc
            train_loss += p[i] * loss
            i += 1

        return train_loss, train_acc    
    
    def save_model(self):
        print("Saving model...")
        if not os.path.exists(self.w_dir):
            os.makedirs(self.w_dir)
        torch.save(self.model.state_dict(), self.w_dir + self.w_file)
        print("Model saved!")
        
    def save_assignment(self, rd = False):
        print("Saving assignment...")
        if not os.path.exists(self.w_dir):
            os.makedirs(self.w_dir)
        #fname =
        #pd.to_csv()
        print("Model saved!")
        
    
    
    def run_train(self):
        # load and distribute data to clients
        print("Creating data distribution for clients...")
        if (self.mode == "non_iid"):
            if (self.zipfz == None):
                self.train_data = self.data_loader.prepare_non_iid_data_option1(len(self.clients))
            else:
                self.train_data = self.data_loader.prepare_non_iid_data_option_zipf(len(self.clients), self.zipfz)
        else:
            self.train_data = self.data_loader.prepare_iid_data(len(self.clients))

        print("Distributing data...")
        for client_id, client_data in tqdm(self.train_data.items()):
            self.clients[client_id].data = client_data
            #print("client, num_batch ", client_id, ", ", len(client_data))
        
        print("Loading z assignment from ", self.z_dir + self.z_file)
        self.load_assignment()
        self.train()

    