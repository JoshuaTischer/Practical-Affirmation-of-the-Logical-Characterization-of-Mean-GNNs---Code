import torch
from torch_geometric.data import Data


def _make(edge_index, x):
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


# bipartite 2x2: 4 node(s), 7 edge(s)
g0 = _make(
    edge_index=[[0, 0, 1, 1, 2, 2, 3], [2, 3, 2, 3, 0, 1, 1]],
    x=[[1, 0], [1, 1], [0, 1], [0, 1]],
)

# bipartite 2x2: 4 node(s), 8 edge(s)
g1 = _make(
    edge_index=[[0, 0, 1, 1, 2, 2, 3, 3], [2, 3, 2, 3, 0, 1, 0, 1]],
    x=[[1, 0], [0, 1], [0, 0], [0, 0]],
)

# bipartite 2x2: 4 node(s), 7 edge(s)
g2 = _make(
    edge_index=[[0, 0, 1, 1, 2, 3, 3], [2, 3, 2, 3, 0, 0, 1]],
    x=[[1, 0], [1, 0], [0, 0], [0, 1]],
)

# bipartite 2x2: 4 node(s), 6 edge(s)
g3 = _make(
    edge_index=[[0, 1, 1, 2, 2, 3], [2, 2, 3, 0, 1, 0]],
    x=[[1, 1], [1, 1], [0, 0], [1, 1]],
)

# bipartite 2x2: 4 node(s), 5 edge(s)
g4 = _make(
    edge_index=[[1, 1, 2, 3, 3], [2, 3, 1, 0, 1]],
    x=[[0, 1], [1, 1], [0, 0], [1, 0]],
)

_unique = [g0, g1, g2, g3, g4]
graphs = [_unique[i % len(_unique)] for i in range(886)]
