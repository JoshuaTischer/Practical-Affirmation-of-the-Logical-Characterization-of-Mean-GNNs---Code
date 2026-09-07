import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


def truncated_relu(x, ceiling=1.0):
    return torch.clamp(x, min=0.0, max=ceiling)


ACTIVATIONS = {
    "relu": F.relu,
    "truncated_relu": truncated_relu,
}

AGGREGATIONS = ("mean", "max", "sum")


class GNN(torch.nn.Module):
    def __init__(self, layer_dimension, aggregation="mean", activation="truncated_relu", bias=True):
        super().__init__()
        if len(layer_dimension) < 2:
            raise ValueError("layer_dimensions must have at least 2 entries")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose: {list(ACTIVATIONS)}")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregation '{aggregation}'. Choose: {AGGREGATIONS}")

        self.num_layers = len(layer_dimension) - 1
        self.aggregation = aggregation
        self.activation = ACTIVATIONS[activation]

        input_dimensions = layer_dimension[0]
        layer_dimension = layer_dimension[1]

        # optional input projection if |Σ| != layer_dim
        if input_dimensions != layer_dimension:
            self.input_projection = torch.nn.Linear(input_dimensions, layer_dimension, bias=False)
        else:
            self.input_projection = None

        # matrices C and A
        self.C = torch.nn.Linear(layer_dimension, layer_dimension, bias=bias)  
        self.A = torch.nn.Linear(layer_dimension, layer_dimension, bias=False) 

    def _aggregate(self, x, edge_index, number_nodes):
        source, destination = edge_index
        # message passing
        return scatter(x[destination], source, dim=0, dim_size=number_nodes, reduce=self.aggregation)

    def forward(self, x, edge_index):
        number_nodes = x.size(0)

        if self.input_projection is not None:
            x = self.input_projection(x)

        for i in range(self.num_layers):
            agg = self._aggregate(x, edge_index, number_nodes)
            x = self.C(x) + self.A(agg)
            if i < self.num_layers - 1:
                x = self.activation(x)

        return x
