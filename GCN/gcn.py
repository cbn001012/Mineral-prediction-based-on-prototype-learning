from torch_geometric.nn import global_add_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 64)

        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)

        x = global_add_pool(x, data.batch) # (100,64)
        # x = global_mean_pool(x, data.batch)
        # x = global_max_pool(x, data.batch)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return F.log_softmax(x, dim=0)


