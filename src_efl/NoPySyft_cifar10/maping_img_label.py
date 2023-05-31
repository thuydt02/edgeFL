from tqdm import tqdm

import torch

import numpy as np
import random
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from datasets import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData
import csv

#run in gg colab
#data_DIR = "/content/drive/MyDrive/eFL/data/"
data_DIR = "../../data" # run local

class Data_Loader:
	def __init__(self, dataset, batch_size):

		self.dataset = dataset
		self.batch_size = batch_size

		transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.1307,), (0.3081,))])
		
		self.train_data = self.load_data(train=True, transform = transform)
		print("type of train_data: ", type(self.train_data))
		#print(self.train_data)
		self.test_data = self.load_test(transform)#DataLoader(dataset=self.load_data(train = False, transform = transform), batch_size=self.batch_size, shuffle=True, drop_last=False)
	
	def load_test(self, transform):
		test_data = self.load_data(train=False, transform = transform)
		x_test = test_data.data
		y_test = test_data.target

		num_batch = int(len(x_test) / self.batch_size)
		
		#x_test = self.normalize(x_test)
		batches = []
		#print("xtest.shape: ", x_test.shape)
		#print("ytest.shape: ", y_test.shape)

		
		for i in range(num_batch):
			start = i * self.batch_size
			end = start + self.batch_size

			batch = TensorDataset(x_test[start:end], y_test[start:end])
			batches.append(batch)
		
		#if end < len(x_test):
		#	batches.append(TensorDataset(x_test[end:len(x_test)], y_test[end:len(y_test)]))
		return DataLoader(ConcatDataset(batches), shuffle=True, batch_size=self.batch_size)
		#return DataLoader(TensorDataset(x_test, y_test), shuffle=True, batch_size=self.batch_size, drop_last = False)
	
	def load_data(self, train, transform):
	    
		if self.dataset == "mnist":
		    # from six.moves import urllib
		    # opener = urllib.request.build_opener()
		    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
		    # urllib.request.install_opener(opener)
		    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		    # data = datasets.MNIST(data_DIR, train=train, download=True, transform=transform)
		    # return data

		    mnist_ds = MNIST_truncated(data_DIR, train=train, download=True, transform=transform)
		    #print(mnist_ds.target[0])
		    #exit()
		    mnist_ds.data = self.normalize(mnist_ds.data)
		    return mnist_ds
		

		elif self.dataset == "femnist":
			femnist_ds = FEMNIST(data_DIR, train=train, transform=transform, download=False)
			femnist_ds.target = femnist_ds.target.long()
			return femnist_ds

			#print("data value range: ", np.max(femnist_ds.data), ", ", np.min(femnist_ds.data))
			#print("data size: ", len(femnist_ds.data))
			#print(femnist_ds.data[0])
			#exit()
		elif self.dataset == "cifar100":
			cifa100 = CIFAR100_truncated(data_DIR, train=train, transform=transform, download=True)
			return cifa100

		elif self.dataset == "cifar10":
			transform_train = transforms.Compose([
		    transforms.RandomCrop(32, padding=4),
		    transforms.RandomHorizontalFlip(),
		    transforms.ToTensor()

		    #width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
		    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			if train == True:
				cifa10 = CIFAR10_truncated(data_DIR, train=train, transform=transform_train, download=True)
			else:
				cifa10 = CIFAR10_truncated(data_DIR, train=train, transform=transform, download=True)
			return cifa10


	def normalize(self, x, mean=0.1307, std=0.3081):
		return (x-mean)/std

	def create_img_label_file(self, fname):
		images = self.train_data.data
		labels = self.train_data.target
		n = len(images)
		with open(fname, 'w', encoding='UTF8') as f:
			writer = csv.writer(f)
			for i in range (n):
				writer.writerow([i, labels[i].item()])
		f.close()


Out_dir = "./../../output/cifar10/z_ass/" 
fname = "map_img_label.csv"

cifar = Data_Loader('cifar10', 20)
cifar.create_img_label_file(Out_dir + fname)
		

	