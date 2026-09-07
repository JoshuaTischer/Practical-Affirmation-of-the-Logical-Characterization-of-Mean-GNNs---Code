import torch
from torch_geometric.data import Data


def _make(edge_index, x):
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


# complete bipartite 3x3: 6 node(s), 18 edge(s)
g0 = _make(
    edge_index=[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
    x=[[1, 0], [0, 1], [1, 1], [0, 0], [0, 0], [0, 0]],
)

# complete bipartite 3x3: 6 node(s), 18 edge(s)
g1 = _make(
    edge_index=[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
    x=[[0, 0], [1, 0], [1, 1], [0, 0], [0, 1], [0, 1]],
)

# complete bipartite 3x3: 6 node(s), 18 edge(s)
g2 = _make(
    edge_index=[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
    x=[[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1]],
)

# complete bipartite 3x3: 6 node(s), 18 edge(s)
g3 = _make(
    edge_index=[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
    x=[[0, 0], [1, 0], [1, 0], [0, 0], [1, 1], [1, 1]],
)

# complete bipartite 3x3: 6 node(s), 18 edge(s)
g4 = _make(
    edge_index=[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
    x=[[0, 0], [0, 0], [1, 1], [0, 0], [0, 1], [1, 1]],
)

# complete bipartite 3x3: 6 node(s), 18 edge(s)
g5 = _make(
    edge_index=[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
    x=[[0, 1], [0, 1], [0, 1], [0, 0], [1, 0], [0, 1]],
)

# complete bipartite 3x3: 6 node(s), 18 edge(s)
g6 = _make(
    edge_index=[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
    x=[[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
)

# complete bipartite 3x3: 6 node(s), 18 edge(s)
g7 = _make(
    edge_index=[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
    x=[[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 1]],
)

# complete bipartite 3x3: 6 node(s), 18 edge(s)
g8 = _make(
    edge_index=[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
    x=[[1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]],
)

# complete bipartite 3x3: 6 node(s), 18 edge(s)
g9 = _make(
    edge_index=[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2]],
    x=[[1, 1], [1, 1], [1, 1], [0, 1], [1, 1], [1, 1]],
)

_unique = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]
graphs = [_unique[i % len(_unique)] for i in range(886)]
