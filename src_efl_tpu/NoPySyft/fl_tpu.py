import os
import torch
import copy
from tqdm import tqdm

from torch import nn

from mnist_model import CNNModel
from client_node import ClientNode
from data_loader import MNISTDataLoader


import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

#import torch_xla
#import torch_xla.core.xla_model as xm

import numpy as np

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SERIAL_EXEC = xmp.MpSerialExecutor()
WRAPPED_MODEL = xmp.MpModelWrapper(CNNModel())

device = xm.xla_device()

torch.manual_seed(1)
torch.set_default_tensor_type('torch.FloatTensor')

class Trainer:
	def __init__(self, no_clients, learning_rate, batch_size, epochs, no_local_epochs, w_dir, acc_dir, mode, zipf_z):
		self.model = WRAPPED_MODEL.to(device) #CNNModel().to(device)
		self.learning_rate = learning_rate * xm.xrt_world_size() # learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.data_loader = MNISTDataLoader(batch_size, device)
		self.clients = self.generate_clients(no_clients)
		self.no_local_epochs = no_local_epochs
		self.w_dir = w_dir
		
		self.mode = mode
		self.zipfz = zipf_z

		self.acc_dir = acc_dir
		self.w_file = "FL_" + mode 
		
		if zipf_z != None:
			self.w_file = self.w_file + "_zipfz" + str(self.zipfz)

		self.w_file = self.w_file + "_B" + str(batch_size) + "_L" + str(no_local_epochs) + "_G" + str(epochs) + ".weight.pth"
		
		self.acc_file = self.w_file + ".acc.csv"

	def generate_clients(self, no_clients):
		clients = [] 
		for i in range(no_clients):
			client = ClientNode(self.learning_rate)
			clients.append(client)

		return clients


	def send_model_to_clients(self):
		for client in self.clients:
			client.model["model"] = copy.deepcopy(self.model)


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

	def train(self):

		best_acc = 0

		coefficients = np.asarray([len(cl.data) for cl in self.clients])

		with open(self.acc_dir + self.acc_file, 'w') as the_file:
			the_file.write("global_rounds,train_loss,train_acc,test_loss,test_acc\n")
        
		print("Start training...")

		for epoch in range(self.epochs):
			print(f"Epoch {epoch+1}/{self.epochs}")
			# Send model to all clients
			self.send_model_to_clients()

			# Update local model for several epochs
			print("Local updating on clients")
			for i in tqdm(range(len(self.clients))):
				for epoch in range(self.no_local_epochs):	
					self.clients[i].train(device)

			# Get back and average the local models
			client_models = [client.model["model"] for client in self.clients]
			self.model = self.average_models(client_models, coefficients)

			# Validate new model


			train_loss, train_acc = self.train_stats()
			print("Training statistic: ")
			print("Accuracy ","{0:.4}%".format(train_acc) , " Loss: ", f'{train_loss:.3}')

			test_loss, test_acc = self.validate(load_weight=False)
			
			with open(self.acc_dir + self.acc_file, 'a') as the_file:
				the_file.write(str((epoch+1) * self.no_local_epochs) + "," + str(train_loss) + "," + str(train_acc) + "," + str(test_loss) + "," + str(test_acc)+ "\n")
            
			if test_acc > best_acc:
				best_acc = test_acc
				self.save_model()
		the_file.close()
        


	def train_stats(self):
		train_acc = 0
		train_loss = 0
		s = sum(np.asarray([len(cl.data) for cl in self.clients]))

		p = [len(cl.data)/s for cl in self.clients]
		i = 0
		for cl in self.clients:
			loss, acc = cl.train_stats(device)
			train_acc += p[i] * acc
			train_loss += p[i] * loss
			i += 1

		return train_loss, train_acc 

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
				#images, labels = images.to(device), labels.to(device)
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

	def save_model(self):
		print("Saving model...")
		if not os.path.exists(self.w_dir):
			os.makedirs(self.w_dir)
		torch.save(self.model.state_dict(), self.w_dir + self.w_file)
		print("Model saved!")


	
	def run_train(self):
		# load and distribute data to clients
		print("Traning FL on " + self.mode + ", zipfz = ", self.zipfz)
		print("Creating data distribution for clients...")
		if (self.mode == "non_iid"):
			if (self.zipfz == None):
				self.train_data = self.data_loader.prepare_non_iid_data_option1(len(self.clients))
			else:
				self.train_data = self.data_loader.prepare_non_iid_data_option_zipf(len(self.clients), self.zipfz)
		else:
			self.train_data = self.data_loader.prepare_iid_data(len(self.clients))

		tmp = []

		#train_loop_fn(para_loader.per_device_loader(device))
    	
		print("Distributing data...")
		for client_id, client_data in tqdm(self.train_data.items()):
			para_loader = pl.ParallelLoader(client_data, [device])
			self.clients[client_id].data = para_loader.per_device_loader(device)

			#self.clients[client_id].data = client_data
			tmp.append(len(client_data))
		
		print("client: num_batch ", tmp)	
		
		self.train()
        
