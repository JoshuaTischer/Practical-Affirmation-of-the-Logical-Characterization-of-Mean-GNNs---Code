import torch
from torch_geometric.data import Data


def _make(edge_index, x):
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


# no neighbors: 1 node(s), 0 edge(s)
g0 = _make(
    edge_index=[[], []],
    x=[[0, 0]],
)

# no neighbors: 1 node(s), 0 edge(s)
g1 = _make(
    edge_index=[[], []],
    x=[[1, 0]],
)

# no neighbors: 1 node(s), 0 edge(s)
g2 = _make(
    edge_index=[[], []],
    x=[[0, 1]],
)

# no neighbors: 1 node(s), 0 edge(s)
g3 = _make(
    edge_index=[[], []],
    x=[[1, 1]],
)

_unique = [g0, g1, g2, g3]
graphs = [_unique[i % len(_unique)] for i in range(886)]
