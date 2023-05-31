import argparse

from eFL import eFL
#from federated_learning import Trainer

NUM_EDGE_SERVERS = 25
NUM_CLIENTS = 1000
BATCH_SIZE = 10
LEARNING_RATE = 0.01 #should be 0.1 for training , 0.01 for pre-training for metis graph
LR_DECAY = 0.995

GLOBAL_ROUNDS = 500000 # global for-loop interations 500 for FL

# number of local client updates before sending model up to server (in FL) or to edge (in eFL)
NUM_LOCAL_UPDATES_BEFORE_SENDING_UP = 123#26#41#205 #26

# the following parameters are used for Edge FL
NUM_EDGE_UPDATES_BEFORE_SENDING_UP = 2 * 123 # 26 * 2 #41*2#205 * 2 #52 = 


# this is needed to compute client-client distance based on FL local weights (after certain iterations)
LOCAL_EPOCHS_PRE_TRAIN = 50

DATASET = "femnist"
MODE = "non_iid"
ZIPF_Z = .7

OUT_DIR = "/content/drive/MyDrive/eFL/output/"
OUT_DIR = "./output/"

w_DIR ="/content/drive/MyDrive/eFL/weight_dirichlet/"
z_DIR ="/content/drive/MyDrive/eFL/z_ass_dirichlet/"
acc_DIR ="/content/drive/MyDrive/eFL/acc_dirichlet/"

w_DIR = OUT_DIR + DATASET + "/weight/"
z_DIR = OUT_DIR + DATASET + "/z_ass/"
acc_DIR = OUT_DIR + DATASET + "/acc/"


#z_FILE = "1z_rnd.part5" #z_metis

z_FILE = "1z_rnd_num_client1000.part25" #z_metis
#z_FILE = "g_no_vw_knn5_d_euclidean_CNN_P1.6m_B10_L20_non_iid_zipfz0.7.part.8"
#z_FILE = "g_d_euclidean_partition_noniid-labeldir0.175_nclient1000.npy.part.35"
#z_FILE = "g_nw_q_d_euclidean_partition_noniid-labeldir0.175_nclient1000.npy.part.25"
q_FILE = "q_lognorm_part1000.npy" 

PRE_TRAINED_W_FILE = "eFL_non_iid_B10_L205_E2_G10000_MNIST_zipfz0.7_g_d_euclidean_L50_non_iid_zipfz0.7.part.5.weight.pth"
PRE_TRAINED_W_FILE = "eFL_MLP2_B10_L40_E2_G500000_g_d_euclidean_partition_noniid-labeldir0.175_nclient1000.npy.part.25.weight.pth"
#PARTITION_DATA_FILE = "/content/drive/MyDrive/eFL/NIID-Bench-main/output/mnist_dir_beta_0.5.npy"
#PARTITION_DATA_FILE = "/content/drive/MyDrive/eFL/NIID-Bench-main/output/partition_mnist_dir_0.175_nparty100npy"
PARTITION_DATA_FILE = "partition_noniid-labeldir0.175_nclient1000.npy"

#w_DIR ="../../weight/" # local
#z_DIR = "../../z_ass/" # local
#acc_DIR = "../../acc/" # local

"""def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', help='Mode: FL-non-iid/eFL-non-iid', default='eFL-non-iid')

    return parser.parse_args()
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', help='Mode: iid (default)/non-iid', default='non_iid')
    
    #parser.add_argument('--num-edge-servers', dest='num_edges', help='Number of edge servers (default 10)', type=int, default=10)
    #parser.add_argument('--num-clients', dest='num_clients', help='Number of client nodes (default 100)', type=int, default=100)
    
    parser.add_argument('--edge-update', dest='edge_update', help='L: Number of epoch to update edge servers models (default 2)', type=int, default=2)
    parser.add_argument('--global-update', dest='global_update', help='E: Number of epoch to update cloud model (default 4)', type=int, default=4)
    
    #parser.add_argument('--epochs', dest='epochs', help='Number of epochs (default 400)', type=int, default=400)
    #parser.add_argument('--batchsize', dest='batchsize', help='Batch size (default 10)', type=int, default=10)
    #parser.add_argument('--lr', dest='lr', help='Learning rate (default 1e-3)', default=1e-3)

    return parser.parse_args()

def main(args):
    if args.mode == 'non_iid':
    
        #print("mode: eFL-non-iid B, L, E: ", BATCH_SIZE, ", ", args.edge_update, ", ", args.global_update)
        #print("Training efl...")
        trainer = eFL(NUM_EDGE_SERVERS, NUM_CLIENTS, GLOBAL_ROUNDS, BATCH_SIZE, LEARNING_RATE, LR_DECAY, args.edge_update, args.global_update, w_DIR, z_DIR, acc_DIR, z_FILE, 
            dataset = DATASET, mode = MODE, pre_trained_w_file = None, zipfz = None, partition_data_file = PARTITION_DATA_FILE, q_file = q_FILE)
        #trainer.pre_train_for_z_graph(LOCAL_EPOCHS_PRE_TRAIN)
        trainer.run_train()

if __name__ == '__main__':
    main(parse_args())



"""import argparse

from cloud_server import CloudServer
from federated_learning import Trainer

NUM_EDGE_SERVERS = 10
NUM_CLIENTS = 100

EDGE_UPDATE_AFTER_EVERY = 2
GLOBAL_UPDATE_AFTER_EVERY = 4

NUM_EPOCHS = 100
BATCH_SIZE = 10
LEARNING_RATE = 1e-3

MODEL_WEIGHT_DIR = "../weight"

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode', help='Mode: federated-non-iid/hierarchical-non-iid', default='hierarchical-non-iid')

	return parser.parse_args()

def main(args):
	if args.mode == 'federated-non-iid':
        print("fedrated-non-iid mode")
        trainer = Trainer(NUM_CLIENTS, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, GLOBAL_UPDATE_AFTER_EVERY, MODEL_WEIGHT_DIR)
        trainer.train()
	else:
        print("hierarchical-non-iid mode")
        trainer = CloudServer(NUM_EDGE_SERVERS, NUM_CLIENTS, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, EDGE_UPDATE_AFTER_EVERY, GLOBAL_UPDATE_AFTER_EVERY, MODEL_WEIGHT_DIR)
#print("abc")
#trainer.pre_train_for_z_graph()
	trainer.train()
#print("ccscsdcsd")
#print("pre-train")
#trainer.pre_train_for_z_graph()
if __name__ == '__main__':
main(parse_args())
"""
