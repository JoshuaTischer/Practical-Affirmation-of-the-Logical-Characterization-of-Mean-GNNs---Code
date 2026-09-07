import torch
from torch_geometric.data import Data


def _make(edge_index, x):
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )

# 3 nodes, 2 edges — uniform star 3 nodes: center [0,0,1], 2 F-leaves → MAX=MEAN=[1,0,0], SUM=2×[1,0,0]
g0 = _make(
    edge_index=[[0, 1, 0, 2], [1, 0, 2, 0]],
    x=[[0, 0, 1], [1, 0, 0], [1, 0, 0]],
)

# 6 nodes, 5 edges — uniform star 6 nodes: center [0,0,1], 5 F-leaves → MAX=MEAN=[1,0,0] same as g0 center; SUM=5×[1,0,0] ≠ g0 (SUM distinguishes!)
g1 = _make(
    edge_index=[[0, 1, 0, 2, 0, 3, 0, 4, 0, 5], [1, 0, 2, 0, 3, 0, 4, 0, 5, 0]],
    x=[[0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
)

# 4 nodes, 3 edges — mixed-leaf star 4 nodes: 1×[1,0,0]+2×[0,0,0] leaves → MAX=[1,0,0]=g0 center, MEAN=[0.33,0,0] ≠ g0 (MEAN distinguishes!)
g2 = _make(
    edge_index=[[0, 1, 0, 2, 0, 3], [1, 0, 2, 0, 3, 0]],
    x=[[0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]],
)

# 5 nodes, 4 edges — path 5 nodes, cycling feature palette
g3 = _make(
    edge_index=[[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
    x=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]],
)

# 4 nodes, 4 edges — cycle 4 nodes, cycling features
g4 = _make(
    edge_index=[[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
    x=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
)

# 10 nodes, 9 edges — caterpillar 10 nodes: spine 2 (alt [0,1,0]/[1,1,0]), 4 pendants/node (alt [1,0,1]/[0,1,1])
g5 = _make(
    edge_index=[[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 6, 1, 7, 1, 8, 1, 9], [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 1, 7, 1, 8, 1, 9, 1]],
    x=[[0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1]],
)

# 8 nodes, 6 edges — two disconnected uniform stars 2+6=8 nodes: center_A: 1 F-leaves, center_B: 5 F-leaves → MAX=MEAN=[1,0,0] for BOTH centers (indistinguishable by MAX/MEAN!), SUM=1×F vs 5×F (SUM distinguishes!)
g6 = _make(
    edge_index=[[0, 1, 2, 3, 2, 4, 2, 5, 2, 6, 2, 7], [1, 0, 3, 2, 4, 2, 5, 2, 6, 2, 7, 2]],
    x=[[0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
)

# 5 nodes, 4 edges — path 5 nodes, alternating [1,0,0]/[0,1,1]: endpoint nodes have 1 heterogeneous neighbor; internal nodes have 2 neighbors with different features
g7 = _make(
    edge_index=[[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
    x=[[1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0]],
)

# 7 nodes, 6 edges — complete binary tree 7 nodes: depth-level features test hierarchical aggregation across MAX/SUM/MEAN
g8 = _make(
    edge_index=[[0, 1, 0, 2, 1, 3, 1, 4, 2, 5, 2, 6], [1, 0, 2, 0, 3, 1, 4, 1, 5, 2, 6, 2]],
    x=[[0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
)

# 8 nodes, 6 edges — two disconnected stars 4+4=8 nodes: center_A sees 3×[1,0,0], center_B sees 3×[0,0,1] → ALL AGGs (MAX, MEAN, SUM) produce different results for both centers
g9 = _make(
    edge_index=[[0, 1, 0, 2, 0, 3, 4, 5, 4, 6, 4, 7], [1, 0, 2, 0, 3, 0, 5, 4, 6, 4, 7, 4]],
    x=[[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
)

graphs = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]
