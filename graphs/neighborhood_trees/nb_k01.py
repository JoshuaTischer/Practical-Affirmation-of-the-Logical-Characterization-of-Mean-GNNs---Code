import torch
from torch_geometric.data import Data


def _make(edge_index, x):
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


# complete bipartite 1x1: 2 node(s), 2 edge(s)
g0 = _make(
    edge_index=[[0, 1], [1, 0]],
    x=[[1, 0], [0, 1]],
)

# complete bipartite 1x1: 2 node(s), 2 edge(s)
g1 = _make(
    edge_index=[[0, 1], [1, 0]],
    x=[[0, 0], [1, 1]],
)

_unique = [g0, g1]
graphs = [_unique[i % len(_unique)] for i in range(886)]
