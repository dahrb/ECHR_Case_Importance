#Adapted from pygeometric example - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

#Version history
#v1_0 = sets up the TGN model and ensures it works with Wiki dataset as baseline
#v1_1 = made changes to the model to accomodate previous timestep embeddings generated from the memory
#v1_2 = adapted the model to include the node features in the graph attention embedding
#v1_3 = implemented the correct ECHR dataset
#v1_4 = completed code for training and testing the model

# This code achieves a performance of around 96.60%. However, it is not
# directly comparable to the results reported by the TGN paper since a
# slightly different evaluation setup is used here.
# In particular, predictions in the same batch are made in parallel, i.e.
# predictions for interactions later in the batch have no access to any
# information whatsoever about previous interactions in the same batch.
# On the contrary, when sampling node neighborhoods for interactions later in
# the batch, the TGN paper code has access to previous interactions in the
# batch.
# While both approaches are correct, together with the authors of the paper we
# decided to present this version here as it is more realsitic and a better
# test bed for future methods.

import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.data import TemporalData
import random
import numpy as np

from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from echr.graph.data_process_KG import ECHRData, DATA
import argparse
import os

# Set environment variable
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPUs are available")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42


def set_seed(seed=SEED):
    """Global seed for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#default settings
memory_dim = 100
time_dim = 100
embedding_dim = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 5
NEIGHBORS = 10
EPOCHS = 20
N_RUNS = 1
NODE_FEATURES_INC = False
VAL_TEST = False

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, node_feat, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        #print('conv: ',node_feat.size())
        if NODE_FEATURES_INC:
            x = torch.cat([x, node_feat], dim=1)
        #print(x.size(1))
        #print(self.conv.in_channels)
            assert x.size(1) == self.conv.in_channels, f"Dimension mismatch: x.size(1)={x.size(1)}, conv.in_channels={self.conv.in_channels}"

        return self.conv(x, edge_index, edge_attr)
    

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


class Model():

    def __init__(self, article='3'):

        # Load the dataset
        self.article = str(article)
        self.dataset = ECHRData(article=self.article)
        self.data = self.dataset[0]

        self.num_nodes = self.data.num_nodes

        # For small datasets, we can put the whole dataset on GPU and thus avoid
        # expensive memory transfer costs for mini-batches:
        self.data = self.data.to(device)

        self.node_features = self.data.x
        
        # Define edge features (e.g., attributes of interactions)
        self.edge_features = self.data.msg#.to(device)
        self.num_node_features = self.node_features.size(1)


        global IN_CHANNELS

        if NODE_FEATURES_INC:
            IN_CHANNELS = memory_dim+self.num_node_features
        else:
            IN_CHANNELS = memory_dim

    def load_data(self):
       
        #ensure split is time-based
        if VAL_TEST:
            self.train_data, self.val_data, self.test_data = self.data.train_val_test_split(
                val_ratio=0.15, test_ratio=0.15)

            self.train_loader = TemporalDataLoader(
                self.train_data,
                batch_size=BATCH_SIZE,
                neg_sampling_ratio=1.0,
            )
            self.val_loader = TemporalDataLoader(
                self.val_data,
                batch_size=BATCH_SIZE,
                neg_sampling_ratio=1.0,
            )
            self.test_loader = TemporalDataLoader(
                self.test_data,
                batch_size=BATCH_SIZE,
                neg_sampling_ratio=1.0,
            )
        else:
            self.train_data = self.data
            self.train_loader = TemporalDataLoader(
                self.train_data,
                batch_size=BATCH_SIZE,
                neg_sampling_ratio=1.0,
            )

        self.neighbor_loader = LastNeighborLoader(self.data.num_nodes, size=NEIGHBORS, device=device)

        self.memory = TGNMemory(
            self.data.num_nodes,
            self.data.msg.size(-1),
            memory_dim,
            time_dim,
            message_module=IdentityMessage(self.data.msg.size(-1), memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)

        self.gnn = GraphAttentionEmbedding(
            in_channels=IN_CHANNELS,
            out_channels=embedding_dim,
            msg_dim=self.data.msg.size(-1),
            time_enc=self.memory.time_enc
        ).to(device)

        self.link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

        self.optimizer = torch.optim.Adam(
            set(self.memory.parameters()) | set(self.gnn.parameters())
            | set(self.link_pred.parameters()), lr=LEARNING_RATE)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Helper vector to map global node indices to local ones.
        self.assoc = torch.empty(self.data.num_nodes, dtype=torch.long, device=device)

    def train(self):
        self.memory.train()
        self.gnn.train()
        self.link_pred.train()

        self.memory.reset_state()  # Start with a fresh memory.
        self.neighbor_loader.reset_state()  # Start with an empty graph.

        total_loss = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(device)  

            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)

            self.assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z_memory, last_update = self.memory(n_id)

            # Check dimensions before performing the convolution
            #assert z_memory.size(1) == memory_dim, f"Dimension mismatch: z_memory.size(1)={z_memory.size(1)}, memory_dim={memory_dim}"
            #print('node_features: ',node_features[n_id].size(), num_node_features)

            #assert node_features[n_id].size(1) == num_node_features, f"Dimension mismatch: node_features[n_id].size(1)={node_features[n_id].size(1)}, num_node_features={num_node_features}"
            z = self.gnn(z_memory, self.node_features[n_id-1], last_update, edge_index, self.data.t[e_id].to(device),
                    self.data.msg[e_id].to(device))
            
            pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
            neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

            loss = self.criterion(pos_out, torch.ones_like(pos_out))
            loss += self.criterion(neg_out, torch.zeros_like(neg_out))

            # Update memory and neighbor loader with ground-truth state.
            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            self.neighbor_loader.insert(batch.src, batch.dst)

            loss.backward()
            self.optimizer.step()
            self.memory.detach()
            total_loss += float( loss) * batch.num_events

        return total_loss / self.train_data.num_events


    @torch.no_grad()
    def test(self,loader):
        self.memory.eval()
        self.gnn.eval()
        self.link_pred.eval()

        torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

        aps, aucs = [], []
        for batch in loader:
            batch = batch.to(device)

            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z_memory, last_update = self.memory(n_id)

            z = self.gnn(z_memory, self.node_features[n_id-1], last_update, edge_index, self.data.t[e_id].to(device),
                    self.data.msg[e_id].to(device))
            pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
            neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pos_out.size(0)),
                torch.zeros(neg_out.size(0))], dim=0)

            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))

            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            self.neighbor_loader.insert(batch.src, batch.dst)

        return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())

    @torch.no_grad()
    def get_graph_at_time(self, timestep=None):

        global NODE_FEATURES_INC
        
        self.memory.eval()
        self.gnn.eval()

        if timestep is not None:
            # Filter the data to include only events up to the specified timestep.
            # TemporalData stores endpoints as src/dst (there is no edge_index).
            mask = self.data.t < timestep
            filtered_src = self.data.src[mask]
            filtered_dst = self.data.dst[mask]
            filtered_t = self.data.t[mask]
            filtered_msg = self.data.msg[mask]
        else:
            # Use the full data
            filtered_src = self.data.src
            filtered_dst = self.data.dst
            filtered_t = self.data.t
            filtered_msg = self.data.msg

        data = TemporalData(src=filtered_src, dst=filtered_dst, t=filtered_t, msg=filtered_msg)

        
        data_loader = TemporalDataLoader(
                data,
                batch_size=5,
                neg_sampling_ratio=0,
            )

        # Initialize the neighbor loader and memory
        self.neighbor_loader.reset_state()
        self.memory.reset_state()

        # Process the filtered data
        #for i in range(filtered_data.size(1)):
            #src, dst = filtered_data[:, i]
            #t = filtered_t[i]
            #msg = filtered_msg[i]
            #src, dst,t,msg = src.unsqueeze(0), dst.unsqueeze(0), t.unsqueeze(0), msg.unsqueeze(0)
            # Update memory and neighbor loader with ground-truth state
            
        for i in data_loader:
            i = i.to(device) 
            self.memory.update_state(i.src, i.dst, i.t, i.msg)
            self.neighbor_loader.insert(i.src, i.dst)

    def get_node_embedding_at_time(self, node_id, node_feat=True):
        
        n_id = torch.tensor([node_id], device=device)
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        z_memory, last_update = self.memory(n_id)
        node_features_here = self.node_features[n_id-1]

        if node_feat:
            z = self.gnn(z_memory, node_features_here, last_update, edge_index, self.data.t[e_id].to(device), self.edge_features[e_id].to(device))
        else:
            node_no = torch.zeros_like(node_features_here)
            z = self.gnn(z_memory, node_no, last_update, edge_index, self.data.t[e_id].to(device), self.edge_features[e_id].to(device))

        #CHECK THIS FOR NUMERICAL INCONSISTENCIES
        node_idx = (n_id == node_id).nonzero(as_tuple=True)[0].item()

        return z[node_idx].unsqueeze(0)  # Return the embedding for the specified node
    
    def load_parameters(self, checkpoint):
        self.gnn.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory.load_state_dict(checkpoint['memory_state_dict'])
        self.link_pred.load_state_dict(checkpoint['link_pred_state_dict'])

def checkpoint_path(model):
    out_dir = os.path.join(DATA, "network", f"article{model.article}")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"best_model_{NODE_FEATURES_INC}.pth")


def main_train_val(model):
    avg_val_ap = 0
    avg_val_auc = 0
    avg_test_ap = 0
    avg_test_auc = 0
    avg_epoch = 0

    save_path = checkpoint_path(model)

    for n_runs in range(N_RUNS):

        best_val_ap = 0

        for epoch in range(1, EPOCHS+1):
            loss = model.train()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
            val_ap, val_auc = model.test(model.val_loader)
            test_ap, test_auc = model.test(model.test_loader)
            #modify for train-test AUC curve?
            print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
            print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')

            # Save the model if it achieves the best validation AP
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.gnn.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'memory_state_dict': model.memory.state_dict(),
                    'link_pred_state_dict': model.link_pred.state_dict(),
                    'loss': loss,
                    'val_ap': val_ap,
                    'val_auc': val_auc,
                    'test_ap': test_ap,
                    'test_auc': test_auc
                }

                torch.save(best_model_state, save_path)
                print(f'Saved best model at epoch {epoch} with Val AP: {val_ap:.4f} -> {save_path}')

        avg_epoch += best_model_state['epoch']
        avg_val_ap += best_model_state['val_ap']
        avg_val_auc += best_model_state['val_auc']
        avg_test_ap += best_model_state['test_ap']
        avg_test_auc += best_model_state['test_auc']

    print(f'Average Val AP: {avg_val_ap/N_RUNS:.4f}, Average Val AUC: {avg_val_auc/N_RUNS:.4f}')
    print(f'Average Test AP: {avg_test_ap/N_RUNS:.4f}, Average Test AUC: {avg_test_auc/N_RUNS:.4f}')
    print(f'Average Epoch: {avg_epoch/N_RUNS:.4f}')

def main_train(model):

    for epoch in range(1, EPOCHS+1):
        loss = model.train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    
    best_model_state = {
        'epoch': epoch,
        'model_state_dict': model.gnn.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'memory_state_dict': model.memory.state_dict(),
        'link_pred_state_dict': model.link_pred.state_dict(),
        'loss': loss}

    save_path = checkpoint_path(model)
    torch.save(best_model_state, save_path)
    print(f'Saved best model at epoch {epoch} -> {save_path}')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TGN model for ECHR dataset')
    parser.add_argument('--article', default='3', help='Article number: 3, 6, or 8')
    parser.add_argument('--node_features_inc', action='store_true', help='Include node features in the graph attention embedding')
    parser.add_argument('--no_node_features_inc', action='store_false', dest='node_features_inc', help='Do not include node features in the graph attention embedding')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--val_test', action='store_true', help='Size of test/ val split')
    parser.add_argument('--no_val_test', action='store_false', dest='val_test', help='Size of test/ val split')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--seed', type=int, default=SEED, help='Global random seed')
    parser.set_defaults(node_features_inc=True)
    parser.set_defaults(val_test=True)
    args = parser.parse_args()

    set_seed(args.seed)

    EPOCHS = args.epochs
    N_RUNS = args.n_runs
    NODE_FEATURES_INC = args.node_features_inc
    IN_CHANNELS = memory_dim
    VAL_TEST = args.val_test
    print(f'[art {args.article}] node_features_inc={NODE_FEATURES_INC} val_test={VAL_TEST} '
          f'epochs={EPOCHS} n_runs={N_RUNS} seed={args.seed}')

    model = Model(article=args.article)

    model.load_data()

    if VAL_TEST:
        main_train_val(model)
    else:
        main_train(model)


