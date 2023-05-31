import torch
import numpy as np
import random

random.seed(1)

class EdgeServerNode:
	def __init__(self, model):
		self.model = None
		self.connected_clients = []
		self.location = np.array((random.random(), random.random()))
		self.data_size = []
		self.list_labels = []

	def add_client(self, client_id):
		self.connected_clients.append(client_id)