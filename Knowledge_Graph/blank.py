from data_process_KG import ECHRData
import torch
from torch_geometric.datasets import JODIEDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#path = '/users/sgdbareh/volatile/ECHR_Importance/Knowledge_Graph/'

dataset = ECHRData(root='.')
#dataset = JODIEDataset(path, name='wikipedia')
# Pad data.msg with zeros to ensure the last dimension size is 5

data = dataset[0]

# if data.msg.size(-1) < 5:
#     padding_size = 5 - data.msg.size(-1)
#     padding = torch.zeros(data.msg.size(0), padding_size, device=device)
#     data.msg = torch.cat([data.msg, padding], dim=-1,device=device)

# #data = data.to(device)

# # Generate random node features for the dataset
num_nodes = data.num_nodes
node_features = data.x
num_node_features = node_features.size(1)

print(node_features[0])

print(data.src) #Expected size 5 but got size 1 - size 0 for mine
