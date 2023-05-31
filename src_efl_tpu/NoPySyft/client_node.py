import random
import numpy as np
import copy
import torch

from mnist_model import CNNModel
from torch import nn, optim


import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl


#import torch_xla
#import torch_xla.core.xla_model as xm

random.seed(1)
np.random.seed(1)

class ClientNode:
	def __init__(self, learning_rate):
		self.model = {"model": None, "optim": None, "criterion": None, "loss": None}
		self.data = []
		self.location = self.generate_location()
		self.learning_rate = learning_rate


	def generate_location(self):
		location_ranges = [0,1,2,3,4]
		distributions = [0.1, 0.15, 0.2, 0.25, 0.3]

		index = np.random.choice(location_ranges, 1, replace=False, p=distributions)[0]
		start = 0.2 * index
		end = start + 0.2

		location = (random.uniform(start, end), random.uniform(start, end))

		return location

	def clear_model(self):
		del self.model["model"]
		self.model["model"] = None


	def train(self, device):
		#if isinstance(self.model["model"], list):
		#	if len(self.model["model"]) > 1:
		#		self.model["model"] = self.average_models()
		#	else:
		#		self.model["model"] = self.model["model"][0]
		#print("in clientnode: self.model['model']: ", self.model["model"])


		
		self.model["optim"] = optim.SGD(self.model["model"].parameters(), lr=self.learning_rate)
		self.model["criterion"] = nn.CrossEntropyLoss() 
	
		local_num_epoch = 1
		
		self.model["model"].train()

		for i in range (local_num_epoch):
			for batch_idx, (images, labels) in enumerate(self.data):
				#images, labels = images.to(device), labels.to(device)

				#if (batch_idx+1)%100==0:
				#	print(f"Processed {batch_idx+1}/{len(self.data)} batches")

				self.model["optim"].zero_grad()
				output = self.model["model"].forward(images)
				loss = self.model["criterion"](output, labels)

				loss.backward()
				#self.model["optim"].step()
				xm.optimizer_step(self.model["optim"])
			# self.sum_model()
		
	def train_stats(self, device):

		#print("Traning statistic:...")
		self.model["model"].eval()
		corrects = 0
		loss = 0

		with torch.no_grad():
			for batch_idx, (images, labels) in enumerate(self.data):
				#images, labels = images.to(device), labels.to(device)
				output = self.model["model"](images)
				pred = output.argmax(dim=1)

				corrects += pred.eq(labels.view_as(pred)).sum().item()
				loss += self.model["criterion"](output, labels).item()

		#print("len(client.data[0]): ", len(self.data[0]))
		
		total_train = len(self.data)*batch_size
		accuracy = 100*corrects/total_train
		loss = loss/len(self.data)

		#print("Number of corrects: {}/{}".format(corrects, len(self.data)*batch_size))
		#print("Loss, Accuracy: {}%, {}".format(loss, accuracy))
		#print("-------------------------------------------")

		return loss, accuracy 

	def sum_model(self):
		total = 0
		for name, param in self.model["model"].named_parameters():
			total += torch.sum(param.data)
		print(total)
