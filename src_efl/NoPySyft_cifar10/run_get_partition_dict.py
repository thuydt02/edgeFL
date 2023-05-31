import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
from utils import *

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='../../data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)
    
    return net_dataidx_map



#calling get_partition_data_IID(dataset, datadir, n_parties):

dataset = "cifar10" # "mnist" #"femnist"
partition = "iid"
n_parties = 50
dict_partition = get_partition_data_IID(dataset = dataset, datadir='../../data', n_parties = n_parties)

#calling get_partition_dict

#dataset = "cifar10" # "mnist" #"femnist"
#partition = "iid-diff-quantity"

#partition = "noniid-labeldir"
#partition = "noniid-#label3"
#n_parties = 250
#beta = 0.5
#dict_partition = get_partition_dict(dataset, partition, n_parties, beta = beta)


#calling getpartition_data_82__(dataset, datadir, partition, nparties, dominate_class_ratio = 0.8)
#dataset = "cifar10" #"mnist"
#datadir = "../../data"
#n_parties = 50
#partition = "noniid90"
#dict_partition = partition_data_82__(dataset, datadir, n_parties, dominate_class_ratio = 0.90)


#calling getpartition_data_82(dataset, datadir, partition, nparties, dominate_class_ratio = 0.8)

#dataset = "mnist"
#datadir = "../../data"
#n_parties = 100
#partition = "noniid82_#label2"
#data_size_distribution_file = "../../output/mnist/z_ass/data_size_zipfz0.7.npy" #client_data_size_CNN_P1.6m_B10_non_iidzipfz0.7"
#dict_partition, label_distribution = partition_data_82(dataset, datadir, n_parties, dominate_class_ratio = 0.8, data_size_distribution_file = data_size_distribution_file)

#save to files

data_size = [len(dict_partition[i]) for i in range(n_parties)]
print ("data size (num of images in each party): ")
print(data_size)

path_file = "../../output/" + dataset + "/z_ass/"
partition_file = "partition_" + partition

if (partition == "noniid-labeldir") or (partition == "iid-diff-quantity") :
    partition_file += str(beta)
partition_file = partition_file + "_nclient" + str(n_parties)

data_size_file = "data_size_" + partition_file + ".npy"

with open(path_file + data_size_file, 'wb') as f:
    np.save(f, np.array(data_size))


with open(path_file + partition_file + ".npy", 'wb') as f:
    for i in range(n_parties):
        np.save(f, np.array(dict_partition[i]))


if partition == "noniid82_#label2":
    label_file = "label_" + partition_file + ".npy"
    with open(path_file + label_file, 'wb') as f:
        np.save(f, label_distribution)



print("saved partition, data_size files!")

#print(dict[0])

