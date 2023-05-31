import argparse

from eFL import eFL
#from federated_learning import Trainer

NUM_EDGE_SERVERS = 8
NUM_CLIENTS = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.001


GLOBAL_ROUNDS = 3900 # global for-loop interations 500 for FL

# number of local client updates before sending model up to server (in FL) or to edge (in eFL)
NUM_LOCAL_UPDATES_BEFORE_SENDING_UP = 26 #26

# the following parameters are used for Edge FL
NUM_EDGE_UPDATES_BEFORE_SENDING_UP = 52 #52


# this is needed to compute client-client distance based on FL local weights (after certain iterations)
LOCAL_EPOCHS_PRE_TRAIN = 50

MODE = "non_iid"
ZIPF_Z = .7


w_DIR ="/content/drive/MyDrive/NA-Multiple-Edge-Servers-Federated-Learning/weight/"
z_DIR ="/content/drive/MyDrive/NA-Multiple-Edge-Servers-Federated-Learning/z_ass/"
acc_DIR ="/content/drive/MyDrive/NA-Multiple-Edge-Servers-Federated-Learning/acc/"

z_FILE = "1z_rnd.part5" 

#w_DIR ="../../weight/" # local
#z_DIR = "../../z_ass/" # local
#acc_DIR = "../../acc/" # local

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', help='Mode: FL-non-iid/eFL-non-iid', default='eFL-non-iid')

    return parser.parse_args()

"""def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', help='Mode: federated-non-iid (default)/hierarchical-non-iid', default='federated-non-iid')
    
    parser.add_argument('--num-edge-servers', dest='num_edges', help='Number of edge servers (default 10)', type=int, default=10)
    parser.add_argument('--num-clients', dest='num_clients', help='Number of client nodes (default 100)', type=int, default=100)
    
    parser.add_argument('--edge-update', dest='edge_update', help='Number of epoch to update edge servers models (default 2)', type=int, default=2)
    parser.add_argument('--global-update', dest='global_update', help='Number of epoch to update cloud model (default 4)', type=int, default=4)
    
    parser.add_argument('--epochs', dest='epochs', help='Number of epochs (default 400)', type=int, default=400)
    parser.add_argument('--batchsize', dest='batchsize', help='Batch size (default 10)', type=int, default=10)
    parser.add_argument('--lr', dest='lr', help='Learning rate (default 1e-3)', default=1e-3)

    return parser.parse_args()
"""
def main(args):
    #if args.mode == 'FL-non-iid':
        #trainer = Trainer(NUM_CLIENTS, LEARNING_RATE, BATCH_SIZE, GLOBAL_ROUND, NUM_LOCAL_UPDATES_BEFORE_SENDING_UP, w_DIR, acc_DIR, MODE, ZIPF_Z)
        #trainer.run_train()
    #else:
    print("mode: eFL-non-iid. Training...")
    trainer = eFL(NUM_EDGE_SERVERS, NUM_CLIENTS, GLOBAL_ROUNDS, BATCH_SIZE, LEARNING_RATE, NUM_LOCAL_UPDATES_BEFORE_SENDING_UP, NUM_EDGE_UPDATES_BEFORE_SENDING_UP, w_DIR, z_DIR, acc_DIR, z_FILE, MODE, ZIPF_Z, )
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
