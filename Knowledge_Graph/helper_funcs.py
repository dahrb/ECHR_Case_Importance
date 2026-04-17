import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from data_process_KG import ECHRData
import argparse
import os
import pandas as pd
from network_TGN import *

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPUs are available")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(filepath='/users/sgdbareh/volatile/ECHR_Importance/best_model_True.pth'):
    # Load the best model
    model = Model()
    checkpoint = torch.load(filepath)
    model.load_data()
    model.load_parameters(checkpoint=checkpoint)

    return model


# for _ in range(10):
#     # Retrieve embeddings for a specific node at a specific timestep for analysis
#     node_id = random.randint(0,4440)#torch.randint(0,4440,size=(1,)).to(device)  # Specify the node ID
#     timestep = 10074  # Specify the timestep

#     print(node_id, timestep)
#     start_time = time.time()

#     embedding_feat = get_node_embedding_at_time(node_id, timestep)
#     print(f'Embedding for node {node_id} at timestep {timestep}: {embedding_feat}')

#     embedding_time = time.time()
#     print(f'Time to get embedding with features: {embedding_time - start_time:.4f} seconds')

#     embedding_no_feat = get_node_embedding_at_time(node_id, timestep, node_feat=False)
#     print(f'Embedding for node {node_id} from the full graph: {embedding_no_feat}')

#     embedding_no_feat_time = time.time()
#     print(f'Time to get embedding without features: {embedding_no_feat_time - embedding_time:.4f} seconds')

#     cosine_similarity = torch.nn.functional.cosine_similarity(embedding_feat, embedding_no_feat)
#     print(f'Cosine similarity between embeddings for node {node_id} at timestep {timestep} and full graph: {cosine_similarity}')

#     similarity_time = time.time()
#     print(f'Time to compute cosine similarity: {similarity_time - embedding_no_feat_time:.4f} seconds')

#     print('Total time for one epoch: ',similarity_time - start_time)
