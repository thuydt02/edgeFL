import argparse
from pre_train_for_distance import pre_train

N_CLIENTS = 300
BATCH_SIZE = 10
LEARNING_RATE = 0.01 #should be 0.01 for MNIST
LR_DECAY = 0.995

GLOBAL_ROUNDS = 100


DATASET = "cifar10" #"femnist"#"mnist"#"cifar100"#"femnist"

OUT_DIR = "/content/drive/MyDrive/eFL/output/"


w_DIR = OUT_DIR + DATASET + "/weight/"
z_DIR = OUT_DIR + DATASET + "/z_ass/"
acc_DIR = OUT_DIR + DATASET + "/acc/"
PARTITION_DATA_FILE = "partition_noniid90_label2_nclient300.npy"


#PARTITION_DATA_FILE = "partition_iid-diff-quantity0.5_nclient100.npy"
#PARTITION_DATA_FILE = "partition_noniid82_nclient100.npy"# "partition_noniid-#label3_nclient600.npy" 
#PARTITION_DATA_FILE = "partition_noniid82_#label2_nclient100.npy"

def parse_args():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--distance-metric', dest='distance_metric', help='Name of distance metric: euclidean | minkowski', type=str, default=None)

    parser.add_argument('--p', dest='p', help='if euclidean, p = 2 else inf > p > 0', type=float, default=None)
    
    #parser.add_argument('--epochs', dest='epochs', help='Number of epochs (default 400)', type=int, default=400)
    #parser.add_argument('--batchsize', dest='batchsize', help='Batch size (default 10)', type=int, default=10)
    #parser.add_argument('--lr', dest='lr', help='Learning rate (default 1e-3)', default=1e-3)

    return parser.parse_args()

def main(args):
	pretrain = pre_train(n_clients = N_CLIENTS, global_rounds = GLOBAL_ROUNDS, batch_size = BATCH_SIZE, learning_rate = LEARNING_RATE, lr_decay = LR_DECAY,
		z_dir = z_DIR, w_dir = w_DIR, dataset = DATASET, pre_trained_w_file = None, partition_data_file = PARTITION_DATA_FILE, distance_metric = args.distance_metric, p = args.p)
	pretrain.get_distance_and_weight_files()

if __name__ == '__main__':
    main(parse_args())