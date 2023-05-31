import argparse

from eFL import eFL

#NUM_EDGE_SERVERS = 15
NUM_CLIENTS = 300#300#1000 #300
BATCH_SIZE = 20
#LEARNING_RATE = 0.0025 #0.01 for cifar10 , 0.01 for pre-training for metis graph; 0.01 for mnist
#LR_DECAY = 0.9

GLOBAL_ROUNDS = 3000 # global for-loop interations 500 for FL

DATASET = "cifar10"#"mnist"#"femnist" #"mnist"#"femnist"#"mnist"#"cifar100"#"femnist"

OUT_DIR = "/content/drive/MyDrive/eFL/output/"
OUT_DIR = "./output/"

w_DIR = OUT_DIR + DATASET + "/weight/"
z_DIR = OUT_DIR + DATASET + "/z_ass/"
acc_DIR = OUT_DIR + DATASET + "/acc/"


#z_FILE = "g_nw_q_d_euclidean_partition_noniid-#label2_nclient100.npy.part.15"
#z_FILE = "1z_rnd_num_client100.part10"


#z_FILE = "1z_rnd_num_client500.part20" #z_metis
#z_FILE = "g_no_vw_knn5_d_euclidean_CNN_P1.6m_B10_L20_non_iid_zipfz0.7.part.8"
#z_FILE = "g_d_euclidean_partition_noniid-labeldir0.175_nclient1000.npy.part.35"
#z_FILE = "g_nw_q_d_euclidean_partition_noniid-labeldir0.175_nclient1000.npy.part.25"
#q_FILE = "q_lognorm_part300.npy"
q_FILE = None # for no q case

#PRE_TRAINED_W_FILE = "eFL_CNN3_B10_L10_E2_G500000_1z_rnd_num_client100.part14_q_lognorm_part100.npy.weight.pth"
#PRE_TRAINED_W_FILE = "eFL_CNN2_B10_L10_E2_G500000_g_nw_q_d_euclidean_partition_noniid-#label2_nclient100.npy.part.15.weight.pth"
#PRE_TRAINED_W_FILE = "eFL_sCNN_B10_L10_E2_G500000_g_nw_q_d_euclidean_partition_noniid-#label2_nclient100.npy.part.15_q_lognorm_part100.npy.weight.pth"
#PARTITION_DATA_FILE = "/content/drive/MyDrive/eFL/NIID-Bench-main/output/mnist_dir_beta_0.5.npy"
#PARTITION_DATA_FILE = "/content/drive/MyDrive/eFL/NIID-Bench-main/output/partition_mnist_dir_0.175_nparty100npy"
#PARTITION_DATA_FILE = "partition_noniid-labeldir0.175_nclient1000.npy"
#PARTITION_DATA_FILE = "partition_noniid-#label2_nclient100.npy"
#PARTITION_DATA_FILE = "partition_iid-diff-quantity0.5_nclient100.npy"

#PARTITION_DATA_FILE = "partition_iid-diff-quantity0.175_nclient1000.npy"
#PARTITION_DATA_FILE = "partition_noniid82_nclient100.npy"
#PARTITION_DATA_FILE = "partition_noniid90_nclient300.npy"
#PARTITION_DATA_FILE = "partition_noniid-labeldir0.175_nclient1000.npy"

#PARTITION_DATA_FILE = "partition_noniid90_label2_nclient300.npy" #label2
#PARTITION_DATA_FILE = "partition_noniid90_nclient300.npy" #label1
PARTITION_DATA_FILE = "partition_noniid90_label3_nclient300.npy" #label3

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
    
    parser.add_argument('--num-edge-servers', dest='num_edges', help='Number of edge servers (default 10)', type=int, default=10)
    parser.add_argument('--mode', dest='mode', help='iid or noniid', type=str, default='noniid')
    
    parser.add_argument('--lr', dest='lr', help='LEARNING RATE', type=float, default=0.003)
    parser.add_argument('--lr-decay', dest='lr_decay', help='LEARNING RATE DECAY', type=float, default=0.995)

    #parser.add_argument('--num-clients', dest='num_clients', help='Number of client nodes (default 100)', type=int, default=100)
    
    parser.add_argument('--edge-update', dest='edge_update', help='L: Number of epoch to update edge servers models (default 2)', type=int, default=2)
    parser.add_argument('--global-update', dest='global_update', help='E: Number of epoch to update cloud model (default 4)', type=int, default=4)
    parser.add_argument('--z-file', dest='z_file', help='z_file: the assignment file', type=str, default=None)
    
    #parser.add_argument('--epochs', dest='epochs', help='Number of epochs (default 400)', type=int, default=400)
    #parser.add_argument('--batchsize', dest='batchsize', help='Batch size (default 10)', type=int, default=10)
    #parser.add_argument('--lr', dest='lr', help='Learning rate (default 1e-3)', default=1e-3)

    return parser.parse_args()

def main(args):
    trainer = eFL(args.num_edges, args.mode, NUM_CLIENTS, GLOBAL_ROUNDS, BATCH_SIZE, args.lr, args.lr_decay, args.edge_update, args.global_update, w_DIR, z_DIR, acc_DIR, 
        z_file = args.z_file, 
        dataset = DATASET, pre_trained_w_file = None, partition_data_file = PARTITION_DATA_FILE, q_file = q_FILE)
    trainer.run_train()

if __name__ == '__main__':
    main(parse_args())


