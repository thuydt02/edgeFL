import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import pandas as pd
from model import *
from datasets import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData
from math import sqrt

import torch.nn as nn

import torch.optim as optim
import torchvision.utils as vutils
import time
import random
from itertools import combinations

from models.mnist_model import Generator, Discriminator, DHead, QHead
from config import params
import sklearn.datasets as sk
from sklearn.datasets import load_svmlight_file

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir):

    print("load_fmnist!!!")

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(root = datadir, train=True, download=True, transform=transform)
    mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_svhn_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()
    print("load_cifar10_data: len_X_train = ", len(X_train))

    return (X_train, y_train, X_test, y_test)

#----------------------------------------------------------------
#new adding
def load_cifar100_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)
#-----------------------------------------------------------------
def load_celeba_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train =  celeba_train_ds.attr[:,gender_index:gender_index+1].reshape(-1)
    y_test = celeba_test_ds.attr[:,gender_index:gender_index+1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)

def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=True)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=True)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.target, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.target, mnist_test_ds.users_index

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts



def partition_data_82(dataset, datadir, n_parties, dominate_class_ratio = 0.8, data_size_distribution_file = None):
    #each client has 2 labels: one is dominated by dominate_class_ratio, another is minor
    #allow duplication


    with open(data_size_distribution_file, 'rb') as f:
        data_size_dist = np.load(f)


    K = 10 # num_labels
    X_train, y_train = [],[]
    
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    
    
    n_samples = len(X_train)
    

    data_size_dist = data_size_dist * n_samples

    label_ind = [[] for _ in range(K)] #label_ind[0] = [0, 1, 2] means y_train[0:3] has label of 0
    
    num_selected = 0
    
    for i in range(n_samples):
        label_ind[y_train[i]].append(i)
    
    print("labels in size")
    for l in range(K):
        print("label ", l, ": ", len(label_ind[l]))


    net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(n_parties)}
    
    num_sample_per_party = int(n_samples/n_parties)
    num_dominant = int(dominate_class_ratio * num_sample_per_party)
    num_minor = num_sample_per_party - num_dominant 
    
    set_not_selected = set(range(n_samples))
    
    label_distribution = np.ones((n_parties,2)) * (-1)

    for cl in range(n_parties):
        dominant_class = random.sample(range(K), 1)[0]
        minor_class = random.sample(set(range(K)) - set({dominant_class}), 1)[0]
        label_distribution[cl][0], label_distribution[cl][1] = dominant_class, minor_class

        num_dominant = int(dominate_class_ratio * data_size_dist[cl])
        net_dataidx_map[cl] = random.sample(label_ind[dominant_class], num_dominant)
        net_dataidx_map[cl] = np.append(net_dataidx_map[cl], random.sample(label_ind[minor_class], int(data_size_dist[cl] - num_dominant)))

    
    for i in range(n_parties):
        print ("client ", i, ": ", len(net_dataidx_map[i]))

    
    return net_dataidx_map, label_distribution


def partition_data_82__(dataset, datadir, n_parties, dominate_class_ratio = 0.8):
    
    #only work for mnist, cifar10
    #see the RL pp for choosing clients for FL: https://iqua.ece.toronto.edu/papers/hwang-infocom20.pdf
    #each client has 10 labels. one is dominated, others are spread among the rest 9 labels

    K = 10 # num_labels
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)

    n_samples = len(X_train)
    label_ind = [[] for _ in range(K)] #label_ind[0] = [0, 1, 2] means label 0 has three images 0, 1, and 2
    
    num_selected = 0
    
    for i in range(n_samples):
        label_ind[y_train[i]].append(i)
    
    print("labels in size")
    for l in range(K):
        print("label ", l, ": ", len(label_ind[l]))


    net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(n_parties)}
    
    num_sample_per_party = int(n_samples/n_parties)
    num_dominant = int(dominate_class_ratio * num_sample_per_party)
    num_minor = num_sample_per_party - num_dominant 
    
    set_not_selected = set(range(n_samples))
    
    cl_start = 0
    num_client_per_label = int (n_parties / K)

    for l in range (K):
        for i in range(num_client_per_label):
            net_dataidx_map[cl_start + i] = label_ind[l][0:num_dominant]
        
            dominant = set(label_ind[l][0:num_dominant])
            label_ind[l] = list(set(label_ind[l]) - dominant)
            set_not_selected = set_not_selected - dominant
        cl_start += num_client_per_label     
    
    for i in range(n_parties):
        minor = random.sample(set_not_selected, num_minor)
        net_dataidx_map[i] = np.append(net_dataidx_map[i], minor)
        set_not_selected = set_not_selected - set(minor)
        
    net_dataidx_map[0] = np.append(net_dataidx_map[0], list(set_not_selected))

    s = []
    for i in range(n_parties):
        s = s + list(net_dataidx_map[i])

    print("num_sample: ", n_samples)
    print("num_sample assigned to ", n_parties, " client: ", len(set(s)))

    print ("The rest after round 1 of assigning dominant:")
    for l in range(K):
        print("label ", l, " ", len(label_ind[l]))

    #print (net_dataidx_map[0])
    return net_dataidx_map

def partition_data_given_num_labels(dataset, datadir, n_parties, num_dominated_labels = 2, dominate_class_ratio = 0.9):
    
    #written for reviwers's asking    
    #only work for mnist, cifar10
    #see the RL pp for choosing clients for FL: https://iqua.ece.toronto.edu/papers/hwang-infocom20.pdf
    #each client has 10 labels: num_dominated_labels are dominated, others are spread among the rest 10 - num_dominated_labels
    #for example, if num_dominated_labels = 2 => each clients will have 2 dominated labels

    K = 10 # num_labels
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)

    n_samples = len(X_train)
    label_ind = [[] for _ in range(K)] #label_ind[0] = [0, 1, 5] means the image #0, #1, and #5 have label 0
    
    num_selected = 0
    
    for i in range(n_samples):
        label_ind[y_train[i]].append(i)

    num_samples_per_label = [len(label_ind[i]) for i in range(K)]
    
    print("labels in size")
    for l in range(K):
        print("label ", l, ": ", len(label_ind[l]))


    
    net_dataidx_map = {i:[] for i in range(n_parties)}
    
    num_sample_per_party = int(n_samples/n_parties) #how many samples for each client

    num_dominant = int(dominate_class_ratio * num_sample_per_party) #how many dominant samples / client
    num_dominant_per_label = int(num_dominant / num_dominated_labels) # how many dominant samples / 1 lable / 1 client 
    
    num_client_per_label = int (n_parties * num_dominated_labels / K) #how many clients / 1 label
    
    max_dominant_per_label = [int(len(label_ind[k]) * dominate_class_ratio) for k in range(K)]
    print ("num_dominant_per_label: ", num_dominant_per_label)

    set_not_selected = set(range(n_samples))
    
    cl_start = 0
    
    label_to_pick = [i for i in range(K)]

    print("max_dominant_per_label: ", max_dominant_per_label)
    #picking dominant
    
    for cl in range(n_parties):
        picked_labels = random.sample(label_to_pick, min(num_dominated_labels, len(label_to_pick)))
        if len(picked_labels) == num_dominated_labels:
            for k in picked_labels:
                net_dataidx_map[cl] += label_ind[k][:num_dominant_per_label]
                label_ind[k] = list(set(label_ind[k]) - set(label_ind[k][:num_dominant_per_label]))
                if len(label_ind[k]) <= num_samples_per_label[k] - max_dominant_per_label[k]:
                    label_to_pick.remove(k)
        else:
            n_picked_samples = 0
            for k in picked_labels:
                while n_picked_samples < num_dominant:
                    net_dataidx_map[cl] += label_ind[k][:num_dominant_per_label]
                    label_ind[k] = list(set(label_ind[k]) - set(label_ind[k][:num_dominant_per_label]))
                    n_picked_samples += num_dominant_per_label
                    if len(label_ind[k]) <= num_samples_per_label[k] - max_dominant_per_label[k]:
                        label_to_pick.remove(k)
                        break



        print ("client, labels picked, total samples picked: ", cl, picked_labels, len(net_dataidx_map[cl]))
    
        #    exit()


    #picking minority

    minority_set = set()
    for k in range(K):
        print("len(label_ind[k]):",k , len(label_ind[k]))
        minority_set = minority_set.union(set(label_ind[k]))

    num_minor = int(len(minority_set) / n_parties)
    
    print("len(minority_set), num_minor: ", len(minority_set), num_minor)
    for i in range(n_parties):
        minor = random.sample(minority_set, num_minor)
        net_dataidx_map[i] += minor #np.append(net_dataidx_map[i], minor)
        minority_set = minority_set - set(minor)
        
    
    print("num_sample: ", n_samples)
    print("num_sample NOT assigned to any clients:",len(minority_set))
    for cl in range(n_parties):
        print ("client, total samples picked: ", cl, len(net_dataidx_map[cl]))
    
    return net_dataidx_map

def partition_data_given_data_size(dataset, datadir, n_parties, dominate_class_ratio = 0.9, data_size_fname = ""):
    
    #written for reviwers's asking    
    #only work for mnist, cifar10
    #distribute the dataset to clients, using the data_size file and each client has one dominated label, the ratio is determined by dominated_class_ratio

    #see the RL pp for choosing clients for FL: https://iqua.ece.toronto.edu/papers/hwang-infocom20.pdf
    #each client has 10 labels: one label is dominated, others are spread among the rest 9 labels
    
    if data_size_fname == "":
        print("No data_size distribution file found")
        exit()
    

    #reading the dataset

    K = 10 # num_labels
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)

    n_samples = len(X_train)
    label_ind = [[] for _ in range(K)] #label_ind[0] = [0, 1, 5] means the image #0, #1, and #5 have label 0    
    for i in range(n_samples):
        label_ind[y_train[i]].append(i)

    num_samples_per_label = [len(label_ind[i]) for i in range(K)]
    print("labels in size")
    for l in range(K):
        print("label ", l, ": ", num_samples_per_label[l])


    #reading the data_size file

    p = pd.read_csv(data_size_fname, header = None, index_col = None) #p is an array of percentages 
    p = p.values.squeeze()

    
    net_dataidx_map = {i:[] for i in range(n_parties)}
    
    
    n_dominated_samples = [int(p[i] * n_samples * dominate_class_ratio) for i in range(n_parties)] #how many dominant samples / client
    
    #num_client_per_label = int (n_parties * num_dominated_labels / K) #how many clients / 1 label
    
    max_dominant_per_label = [int(len(label_ind[k]) * dominate_class_ratio) for k in range(K)]
    #print ("num_dominant_per_label: ", num_dominant_per_label)

    #set_not_selected = set(range(n_samples))
    
    
    label_to_pick = [i for i in range(K)]

    print("max_dominant_per_label: ", max_dominant_per_label)
    
    #picking dominated labels
    
    for cl in range(n_parties):
        
        candidate_labels = [l for l in label_to_pick if len(label_ind[l]) >= n_dominated_samples[cl] + num_samples_per_label[l] - max_dominant_per_label[l]]
        if candidate_labels == []:
            break
        
        k = random.sample(candidate_labels, 1)[0] #picking label k

        net_dataidx_map[cl] = label_ind[k][:n_dominated_samples[cl]]
        label_ind[k] = list(set(label_ind[k]) - set(label_ind[k][:n_dominated_samples[cl]]))
        if len(label_ind[k]) <= num_samples_per_label[k] - max_dominant_per_label[k]:
            label_to_pick.remove(k)
        
        print ("client, labels picked, samples picked: ", cl, k, len(net_dataidx_map[cl]))
    
        #    exit()


    #picking minority

    minority_set = set()
    for k in range(K):
        print("len(label_ind[k]):",k , len(label_ind[k]))
        minority_set = minority_set.union(set(label_ind[k]))

    num_minor = int(len(minority_set) / n_parties)
    
    print("len(minority_set), num_minor: ", len(minority_set), num_minor)
    for i in range(n_parties):
        minor = random.sample(minority_set, num_minor)
        net_dataidx_map[i] += minor #np.append(net_dataidx_map[i], minor)
        minority_set = minority_set - set(minor)
        
    
    #assign the rest of the dataset to clients
    candidate_clients = [c for c in range(n_parties) if len(net_dataidx_map[c]) < 20]
    
    while len(minority_set) > 0:
        for i in candidate_clients:
            minor = random.sample(minority_set, min(20, len(minority_set)))
            net_dataidx_map[i] += minor #np.append(net_dataidx_map[i], minor)
            minority_set = minority_set - set(minor)
            if len(minority_set) == 0:
                break 


    print("num_sample: ", n_samples)
    print("num_sample NOT assigned to any clients:",len(minority_set))

    for cl in range(n_parties):
        print ("client, total samples picked: ", cl, len(net_dataidx_map[cl]))
    
    #exit()
    return net_dataidx_map

def partition_data_C1(dataset, datadir, n_parties = 10):
    
    #written for reviwers's asking for an experiment showing that clients with similar data distributions have the models with close distances, and vice versa      
    #only work for mnist, cifar10
    #here we will distribute all images with label i to client i. So we have 10 labels => 10 clients

    K = 10 # num_labels
    n_parties = K

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)

    n_samples = len(X_train)
    label_ind = [[] for _ in range(K)] #label_ind[0] = [0, 1, 5] means the image #0, #1, and #5 have label 0
    
    num_selected = 0
    
    for i in range(n_samples):
        label_ind[y_train[i]].append(i)

    num_samples_per_label = [len(label_ind[i]) for i in range(K)]
    
    print("labels in size")
    for l in range(K):
        print("label ", l, ": ", len(label_ind[l]))

    
    net_dataidx_map = {i:label_ind[i] for i in range(n_parties)}
    
    return net_dataidx_map



def get_partition_data_IID(dataset, datadir, n_parties):
    K = 10 # num_labels
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        K = 100
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)

    n_samples = len(X_train)
    label_ind = [[] for _ in range(K)] #label_ind[0] = [0, 1, 2] means y_train[0:3] has label of 0
    
    for i in range(n_samples):
        label_ind[y_train[i]].append(i)
    
    print("labels in size")
    for l in range(K):
        print("label ", l, ": ", len(label_ind[l]))


    net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(n_parties)}
    
    for l in range(K):
        num_sample_per_party = int(len(label_ind[l]) / n_parties)
        for i in range(n_parties):
            picked = random.sample(label_ind[l], num_sample_per_party)
            net_dataidx_map[i] = np.append(net_dataidx_map[i], picked)
            label_ind[l] = list(set(label_ind[l]) - set(picked))


    
    the_rest = []
    for l in range(K):
        the_rest = the_rest + label_ind[l]
    u = int(len(the_rest)/n_parties)
    
    for i in range(n_parties):
        picked = random.sample(the_rest, u)
        net_dataidx_map[i] = np.append(net_dataidx_map[i], picked)
        the_rest = list(set(the_rest) - set(picked))

    print ("The rest after distributing to all parties: n_samples = ", len(the_rest))
    
    print("Data size distribution across clients: ")
    for i in range (n_parties):
        print('client ', i, ': ', len(net_dataidx_map[i]))
    return net_dataidx_map



def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    #np.random.seed(2020)
    #torch.manual_seed(2020)
    #X_train, y_train, X_text, y_test = [], [], [], []


    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
    elif dataset == 'generated':
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 ==1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1>0:
                y_test.append(0)
            else:
                y_test.append(1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int64)
        idxs = np.linspace(0,3999,4000,dtype=np.int64)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy",X_train)
        np.save("data/generated/X_test.npy",X_test)
        np.save("data/generated/y_train.npy",y_train)
        np.save("data/generated/y_test.npy",y_test)
    
    #elif dataset == 'covtype':
    #    cov_type = sk.fetch_covtype('./data')
    #    num_train = int(581012 * 0.75)
    #    idxs = np.random.permutation(581012)
    #    X_train = np.array(cov_type['data'][idxs[:num_train]], dtype=np.float32)
    #    y_train = np.array(cov_type['target'][idxs[:num_train]], dtype=np.int32) - 1
    #    X_test = np.array(cov_type['data'][idxs[num_train:]], dtype=np.float32)
    #    y_test = np.array(cov_type['target'][idxs[num_train:]], dtype=np.int32) - 1
    #    mkdirs("data/generated/")
    #    np.save("data/generated/X_train.npy",X_train)
    #    np.save("data/generated/X_test.npy",X_test)
    #    np.save("data/generated/y_train.npy",y_train)
    #    np.save("data/generated/y_test.npy",y_test)

    elif dataset in ('rcv1', 'SUSY', 'covtype'):
        X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
        X_train = X_train.todense()
        num_train = int(X_train.shape[0] * 0.75)
        if dataset == 'covtype':
            y_train = y_train-1
        else:
            y_train = (y_train+1)/2
        idxs = np.random.permutation(X_train.shape[0])

        X_test = np.array(X_train[idxs[num_train:]], dtype=np.float32)
        y_test = np.array(y_train[idxs[num_train:]], dtype=np.int32)
        X_train = np.array(X_train[idxs[:num_train]], dtype=np.float32)
        y_train = np.array(y_train[idxs[:num_train]], dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy",X_train)
        np.save("data/generated/X_test.npy",X_test)
        np.save("data/generated/y_train.npy",y_train)
        np.save("data/generated/y_test.npy",y_test)

    elif dataset in ('a9a'):
        X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
        X_test, y_test = load_svmlight_file("../../../data/{}.t".format(dataset))
        X_train = X_train.todense()
        X_test = X_test.todense()
        X_test = np.c_[X_test, np.zeros((len(y_test), X_train.shape[1] - np.size(X_test[0, :])))]

        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = (y_train+1)/2
        y_test = (y_test+1)/2
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy",X_train)
        np.save("data/generated/X_test.npy",X_test)
        np.save("data/generated/y_train.npy",y_train)
        np.save("data/generated/y_test.npy",y_test)


    n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100

        N = y_train.shape[0]
        np.random.seed(2020)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:]) # number of labels for each client
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2 # number of labels in the whole dataset
        #elif dataset in ['cifar10', 'femnist', 'mnist']:
        #    K = 10
        elif dataset == 'cifar100':
            K = 100
        else:
            K = 10
            
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(K)]
            contain=[]
            for i in range(n_parties):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "real" and dataset == "femnist":
        num_user = u_train.shape[0]
        user = np.zeros(num_user+1,dtype=np.int32)
        for i in range(1,num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, n_parties)
        net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(n_parties)}
        for i in range(n_parties):
            for j in batch_idxs[i]:
                net_dataidx_map[i]=np.append(net_dataidx_map[i], np.arange(user[j], user[j+1]))

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    # logger.info("net.parameter.data:", list(net.parameters()))
    paramlist=list(trainable)
    N=0
    for params in paramlist:
        N+=params.numel()
        # logger.info("params.data:", params.data)
    X=torch.empty(N,dtype=torch.float64)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel
    # logger.info("get trainable x:", X)
    return X


def put_trainable_parameters(net,X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0):
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10', 'svhn', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'femnist':
            dl_obj = FEMNIST
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'svhn':
            dl_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])


        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        else:
            dl_obj = Generated
            transform_train = None
            transform_test = None


        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        c_tmp = np.array(choice)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(len(choice), size=batch_size)
            for j in range(batch_size):
                idx[i][j] = c_tmp[int(idx[i][j])]

            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx


