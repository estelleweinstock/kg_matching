import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, RGATConv


class GATClassic(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1_one = GATConv(num_node_features, num_node_features)
        #self.conv2_one = GATConv(num_node_features, num_node_features)

        self.conv1_two = GATConv(num_node_features, num_node_features)
        #self.conv2_two = GATConv(num_node_features, num_node_features)

    def forward(self, data):

        x_one, edge_index_one, edge_attr_one = data['x_one'], data['edge_index_one'], torch.tensor(data['edge_attr_emb_one'], dtype=torch.float)
        x_one = self.conv1_one(x_one, edge_index_one, edge_attr_one).relu()
        x_one = F.dropout(x_one, training=self.training)
        #x_one = self.conv2_one(x_one, edge_index_one, edge_attr_one).relu()

        x_two, edge_index_two, edge_attr_two = data['x_two'], data['edge_index_two'], torch.tensor(data['edge_attr_emb_two'], dtype=torch.float)
        x_two = self.conv1_two(x_two, edge_index_two, edge_attr_two).relu()
        x_two = F.dropout(x_two, training=self.training)
        #x_two = self.conv2_two(x_two, edge_index_two, edge_attr_two).relu()
        return x_one, x_two


class GCNClassic(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1_one = GCNConv(num_node_features, num_node_features)
        self.conv1_two = GCNConv(num_node_features, num_node_features)

    def forward(self, data):

        x_one, edge_index_one = data['x_one'], data['edge_index_one']
        x_one = self.conv1_one(x_one, edge_index_one).relu()
        x_one = F.dropout(x_one, training=self.training)

        x_two, edge_index_two = data['x_two'], data['edge_index_two']
        x_two = self.conv1_two(x_two, edge_index_two).relu()
        x_two = F.dropout(x_two, training=self.training)
        return x_one, x_two


class RGATClassic(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations):
        super().__init__()

        self.conv1_one = RGATConv(in_channels, hidden_channels, num_relations)
        self.conv2_one = RGATConv(in_channels, hidden_channels, num_relations)

        self.conv1_two = RGATConv(hidden_channels, hidden_channels, num_relations)
        self.conv2_two = RGATConv(hidden_channels, hidden_channels, num_relations)


    def forward(self, data):

        x_one, edge_index_one, edge_type_one = data['x_one'], data['edge_index_one'], data['edge_type_one']
        x_one = self.conv1_one(x_one, edge_index_one, edge_type_one).relu()
        #x_one = self.conv2_one(x_one, edge_index_one, edge_type_one).relu()

        x_two, edge_index_two, edge_type_two = data['x_two'], data['edge_index_two'], data['edge_type_two']
        x_two = self.conv1_two(x_two, edge_index_two, edge_type_two).relu()
        #x_two = self.conv2_two(x_two, edge_index_two, edge_type_two).relu()

        return x_one, x_two
