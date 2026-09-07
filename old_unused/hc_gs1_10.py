import torch
from torch_geometric.data import Data


def _make(edge_index, x):
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


# 3 nodes, 4 edges
g0 = _make(
    edge_index=[[0,0,1,2], 
                [1,2,0,0]],
    x=[[1, 0, 0], [1, 0, 0], [0, 1, 0]],
)

# 3 nodes, 4 edges
g1 = _make(
    edge_index=[[0,0,1,2], 
                [1,2,0,0]],
    x=[[1, 0, 0], [1, 0, 0], [0, 1, 0]],
)

# 2 nodes, 2 edges
g2 = _make(
    edge_index=[[0,1], 
                [1,0]],
    x=[[1, 0, 0], [0, 0, 1]],
)

# 5 nodes, 8 edges
g3 = _make(
    edge_index=[[0,0,0,0,1,2,3,4], 
                [1,2,3,4,0,0,0,0]],
    x=[[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
)

# 8 nodes, 7 edges
g4 = _make(
    edge_index=[[0,1,1,2,3,3,3], 
                [1,2,3,4,5,6,7]],
    x=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
)

# 5 nodes, 4 edges
g5 = _make(
    edge_index=[[0,0,2,3], 
                [1,2,3,4]],
    x=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]],
)

# 4 nodes, 3 edges
g6 = _make(
    edge_index=[[0,0,0], 
                [1,2,3]],
    x=[[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
)

# 3 nodes, 2 edges
g7 = _make(
    edge_index=[[0,0], 
                [1,2]],
    x=[[0, 0, 0], [1, 0, 0], [1, 0, 0]],
)

# 3 nodes, 2 edges
g8 = _make(
    edge_index=[[0,0], 
                [1,2]],
    x=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
)

# 9 nodes, 4 edges
g9 = _make(
    edge_index=[[0,0,0], 
                [1,2,3]],
    x=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
)

graphs = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]
