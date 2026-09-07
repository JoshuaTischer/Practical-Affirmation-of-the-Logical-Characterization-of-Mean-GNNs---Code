import os
import random

# Configuration
NUM_GRAPHS_PER_FILE = 10   
NUM_FEATURES = 3           
EDGE_PROB = 0.1            
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "graphs")

RANGES = [
    (1,  10),
    (11, 20),
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 60),
    (61, 70),
    (71, 80),
    (81, 90),
    (91, 100),
]


def random_graph(num_nodes, edge_prob, num_features):
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if random.random() < edge_prob:
                src += [i]
                dst += [j]

    x = [[random.randint(0, 1) for _ in range(num_features)]
         for _ in range(num_nodes)]

    return src, dst, x


def write_file(filepath, low, high, num_graphs, edge_prob, num_features):
    lines = []
    lines.append("import torch")
    lines.append("from torch_geometric.data import Data")
    lines.append("")
    lines.append("")
    lines.append("def _make(edge_index, x):")
    lines.append("    return Data(")
    lines.append("        x=torch.tensor(x, dtype=torch.float),")
    lines.append("        edge_index=torch.tensor(edge_index, dtype=torch.long),")
    lines.append("    )")
    lines.append("")
    lines.append("")

    graph_names = []
    for i in range(num_graphs):
        num_nodes = random.randint(low, high)
        src, dst, x = random_graph(num_nodes, edge_prob, num_features)
        name = f"g{i}"
        graph_names.append(name)
        lines.append(f"# {num_nodes} nodes, {len(src) // 2} edges")
        lines.append(f"{name} = _make(")
        lines.append(f"    edge_index={[src, dst]},")
        lines.append(f"    x={x},")
        lines.append(")")
        lines.append("")

    lines.append(f"graphs = [{', '.join(graph_names)}]")
    lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def main():
    for low, high in RANGES:
        filename = f"rg_gs{low}_{high}.py"
        filepath = os.path.join(OUTPUT_DIR, filename)
        write_file(filepath, low, high, NUM_GRAPHS_PER_FILE, EDGE_PROB, NUM_FEATURES)
        print(f"Generated {filename}")


if __name__ == "__main__":
    main()
