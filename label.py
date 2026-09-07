import torch

# nodes label of the ith feature
def atom(x, i):
    return x[:, i].long()

# bottom: always false (0 for every node)
def bot(x):
    return torch.zeros(x.shape[0], dtype=torch.long)

# top: always true (1 for every node)
def top(x):
    return torch.ones(x.shape[0], dtype=torch.long)

# negation
def neg(phi):
    return 1 - phi

# Or
def l_or(phi1, phi2):
    return (phi1 | phi2).long()

# and
def l_and(phi1, phi2):
    return (phi1 & phi2).long()

# ML Box
def ml_box(edge_index, phi, num_nodes):
    src, dst = edge_index
    result = torch.ones(num_nodes, dtype=torch.long)  
    not_satisfied = ~phi[dst].bool()
    result[src[not_satisfied]] = 0
    return result

# ML Diamond
def ml_dm(edge_index, phi, num_nodes):
    src, dst = edge_index          
    result = torch.zeros(num_nodes, dtype=torch.long)
    satisfied = phi[dst].bool()
    result[src[satisfied]] = 1
    return result

# GML Diamond
def gml_dm(edge_index, phi, num_nodes, g):
    src, dst = edge_index
    count = torch.zeros(num_nodes, dtype=torch.long)
    count.scatter_add_(0, src, phi[dst])
    return (count >= g).long()

# RML Diamond >=
def rml_dm_geq(edge_index, phi, num_nodes, r):
    src, dst = edge_index
    count = torch.zeros(num_nodes, dtype=torch.float)
    degree = torch.zeros(num_nodes, dtype=torch.float)
    count.scatter_add_(0, src, phi[dst].float())
    degree.scatter_add_(0, src, torch.ones(src.shape[0]))
    ratio = torch.where(degree > 0, count / degree, torch.ones_like(count))
    return (ratio >= r).long()

# RML Diamond >
def rml_dm_g(edge_index, phi, num_nodes, r):
    src, dst = edge_index
    count = torch.zeros(num_nodes, dtype=torch.float)
    degree = torch.zeros(num_nodes, dtype=torch.float)
    count.scatter_add_(0, src, phi[dst].float())
    degree.scatter_add_(0, src, torch.ones(src.shape[0]))
    ratio = torch.where(degree > 0, count / degree, torch.zeros_like(count))
    return (ratio > r).long()

def as_many_as(edge_index, phi_1, phi_2, num_nodes):
    src, dst = edge_index
    count_1 = torch.zeros(num_nodes, dtype=torch.long)
    count_2 = torch.zeros(num_nodes, dtype=torch.long)
    count_1.scatter_add_(0, src, phi_1[dst])
    count_2.scatter_add_(0, src, phi_2[dst])
    return (count_1 == count_2).long()


# evaluate formula on graph
def y(edge_index, phi, num_nodes=None):
    if num_nodes is None:
        if edge_index.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        num_nodes = edge_index.max().item() + 1
    return phi(edge_index, num_nodes)
