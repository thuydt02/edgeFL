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

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)
    
    return net_dataidx_map


dataset = "femnist"
partition = "iid-diff-quantity" #noniid-#label2"

#partition = "noniid-labeldir"

n_parties = 1000
beta = 0.175
dict = get_partition_dict(dataset, partition, n_parties, beta = beta)

data_size = [len(dict[i]) for i in range(n_parties)]

path_file = "../output/" + dataset + "/z_ass/"
partition_file = "partition_" + partition

if (partition == "noniid-labeldir") or (partition == "iid-diff-quantity"):
    partition_file += str(beta)
partition_file = partition_file + "_nclient" + str(n_parties)

data_size_file = "data_size_" + partition_file + ".npy"

with open(path_file + data_size_file, 'wb') as f:
    np.save(f, np.array(data_size))

with open(path_file + partition_file + ".npy", 'wb') as f:
    for i in range(n_parties):
        np.save(f, np.array(dict[i]))

print("saved partition, data_size files!")

"""
dataset = "femnist"
partition = "iid-diff-quantity"

#partition = "noniid-labeldir"
n_parties = 1000
beta = 0.175
dict = get_partition_dict(dataset, partition, n_parties, beta = beta)

data_size = [len(dict[i]) for i in range(n_parties)]
print ("\data size (num of images in each party): ")
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
        np.save(f, np.array(dict[i]))

print("saved partition, data_size files!")

#print(dict[0])

"""
