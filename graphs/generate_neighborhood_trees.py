#!/usr/bin/env python3
"""Generate complete-bipartite neighborhood graphs for GNN testing.

Each graph starts with complete bipartite candidates K_{k,k}: two layers of k
nodes each, with every node in one layer connected to every node in the other
layer (edges in both directions). `EDGE_PROBABILITY` independently retains
each directed candidate edge, defaulting to 1.0 so every node has exactly k
neighbors and none are leaves. 

For a given layer size k, every unordered assignment of feature-types across
the k nodes of one layer is enumerated exactly once (order among the k nodes
doesn't matter, only how many of them have each of the 4 feature types) --
C(k+3, 3) unique layer configurations. Configurations are randomly paired up
two at a time (one leftover configuration, if any, is paired with itself) to
form complete bipartite graphs, so every configuration still appears in at
least one graph.

k=20 yields the maximum: C(23, 3) = 1771 configurations -> ceil(1771 / 2) =
886 graphs (the old star-tree design had 4 * C(23, 3) = 7084 unique trees per
k; dividing by 4 removes the root-feature-type factor that no longer applies,
and dividing by 2 accounts for two configurations combining into one graph).
So that no smaller k is underrepresented, every file's `graphs` list is padded
(by cyclically repeating its unique graphs) up to exactly 886 entries. k=0 has
no neighbors, so there is no bipartite pair to build -- as before, it's just
the 4 single, edge-less nodes (one per feature type), cyclically padded like
every other file.
"""

import os
import random
from math import comb

MAX_NEIGHBORS = 20
FEATURE_TYPES = [(0, 0), (1, 0), (0, 1), (1, 1)]  # none, feat1, feat2, both
RANDOM_SEED = 42
EDGE_PROBABILITY = 0.8

_max_configs = comb(MAX_NEIGHBORS + 3, 3)  # 1771 configurations for k=20
TARGET_COUNT = -(-_max_configs // 2)  # ceil(1771 / 2) = 886

OUT_DIR = os.path.join(os.path.dirname(__file__), "neighborhood_trees")


def compositions(total, parts):
    """All tuples of length `parts` of non-negative ints summing to `total`."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def layer_configs(k):
    """All feature-type assignments (as a list of k feature tuples) for one layer."""
    configs = []
    for counts in compositions(k, len(FEATURE_TYPES)):
        feats = []
        for type_idx, count in enumerate(counts):
            feats.extend([FEATURE_TYPES[type_idx]] * count)
        configs.append(feats)
    return configs


def pair_configs(configs, rng):
    """Randomly pair up configs two at a time; a leftover config is paired with itself."""
    shuffled = configs[:]
    rng.shuffle(shuffled)
    pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled) - 1, 2)]
    if len(shuffled) % 2 == 1:
        pairs.append((shuffled[-1], shuffled[-1]))
    return pairs


def render_file(k, pairs, rng, edge_probability):
    lines = [
        "import torch",
        "from torch_geometric.data import Data",
        "",
        "",
        "def _make(edge_index, x):",
        "    return Data(",
        "        x=torch.tensor(x, dtype=torch.float),",
        "        edge_index=torch.tensor(edge_index, dtype=torch.long),",
        "    )",
        "",
        "",
    ]

    for i, (left_feats, right_feats) in enumerate(pairs):
        x = [list(f) for f in left_feats] + [list(f) for f in right_feats]
        left_idx = list(range(k))
        right_idx = list(range(k, 2 * k))
        # Complete bipartite candidates in both directions; each directed edge is retained independently.
        candidate_edges = [(l, r) for l in left_idx for r in right_idx]
        candidate_edges += [(r, l) for r in right_idx for l in left_idx]
        edges = [edge for edge in candidate_edges if rng.random() < edge_probability]
        sources = [source for source, _ in edges]
        targets = [target for _, target in edges]
        lines.append(f"# bipartite {k}x{k}: {2 * k} node(s), {len(sources)} edge(s)")
        lines.append(f"g{i} = _make(")
        lines.append(f"    edge_index=[{sources}, {targets}],")
        lines.append(f"    x={x},")
        lines.append(")")
        lines.append("")

    unique_names = ", ".join(f"g{i}" for i in range(len(pairs)))
    lines.append(f"_unique = [{unique_names}]")
    lines.append(f"graphs = [_unique[i % len(_unique)] for i in range({TARGET_COUNT})]")
    lines.append("")
    return "\n".join(lines)


def render_zero_file():
    """k=0: no neighbors, so each graph is just a single, edge-less node -- one
    per feature type, cyclically padded like every other file."""
    lines = [
        "import torch",
        "from torch_geometric.data import Data",
        "",
        "",
        "def _make(edge_index, x):",
        "    return Data(",
        "        x=torch.tensor(x, dtype=torch.float),",
        "        edge_index=torch.tensor(edge_index, dtype=torch.long),",
        "    )",
        "",
        "",
    ]

    for i, feat in enumerate(FEATURE_TYPES):
        lines.append("# no neighbors: 1 node(s), 0 edge(s)")
        lines.append(f"g{i} = _make(")
        lines.append("    edge_index=[[], []],")
        lines.append(f"    x={[list(feat)]},")
        lines.append(")")
        lines.append("")

    unique_names = ", ".join(f"g{i}" for i in range(len(FEATURE_TYPES)))
    lines.append(f"_unique = [{unique_names}]")
    lines.append(f"graphs = [_unique[i % len(_unique)] for i in range({TARGET_COUNT})]")
    lines.append("")
    return "\n".join(lines)


def main():
    if not 0.0 <= EDGE_PROBABILITY <= 1.0:
        raise ValueError("EDGE_PROBABILITY must be between 0.0 and 1.0")
    os.makedirs(OUT_DIR, exist_ok=True)
    init_path = os.path.join(OUT_DIR, "__init__.py")
    if not os.path.exists(init_path):
        open(init_path, "w").close()

    rng = random.Random(RANDOM_SEED)
    print(f"Edge probability: {EDGE_PROBABILITY}")
    for k in range(MAX_NEIGHBORS + 1):
        if k == 0:
            content = render_zero_file()
            num_unique = len(FEATURE_TYPES)
        else:
            configs = layer_configs(k)
            assert len(configs) == comb(k + 3, 3)
            pairs = pair_configs(configs, rng)
            content = render_file(k, pairs, rng, EDGE_PROBABILITY)
            num_unique = len(pairs)
        out_path = os.path.join(OUT_DIR, f"nb_k{k:02d}.py")
        with open(out_path, "w") as f:
            f.write(content)
        print(f"k={k:2d}: {num_unique:4d} unique graphs -> {TARGET_COUNT} in graphs list -> {out_path}")


if __name__ == "__main__":
    main()
