import random


adj_str = """
0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0, 
0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0, 
0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0, 
0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0, 
0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0, 
0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0, 
0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1, 
0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1, 
0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  1,  1, 
0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  1, 
1, 1, 1, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
0, 1, 1, 1, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
0, 0, 1, 1, 1, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
0, 0, 0, 1, 1, 1, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
0, 0, 0, 0, 1, 1, 1, 1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
0, 0, 0, 0, 0, 1, 1, 1, 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
0, 0, 0, 0, 0, 0, 1, 1, 1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
1, 0, 0, 0, 0, 0, 0, 1, 1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
1, 1, 0, 0, 0, 0, 0, 0, 1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
1, 1, 1, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
"""

VAR_NAME = "g0"
RANDOM_FEATURES = False   # set False for all-zero features
FEATURE_DIM = 3

values = [int(v.strip()) for v in adj_str.replace('\n', ',').split(',') if v.strip() != '']
n_nodes = int(len(values) ** 0.5)
assert n_nodes * n_nodes == len(values), "Input is not a square matrix"

src, dst = [], []
for i in range(n_nodes):
    for j in range(n_nodes):
        if values[i * n_nodes + j] == 1:
            src.append(i)
            dst.append(j)

n_edges = len(src)

if RANDOM_FEATURES:
    x = [[random.randint(0, 1) for _ in range(FEATURE_DIM)] for _ in range(n_nodes)]
else:
    x = [[0] * FEATURE_DIM] * n_nodes


def fmt_edge_index(src, dst, indent=16):
    pad = " " * indent
    src_str = ", ".join(str(v) for v in src)
    dst_str = ", ".join(str(v) for v in dst)
    return f"[[{src_str}],\n{pad} [{dst_str}]]"

def fmt_x(x, per_line=10, indent=4):
    pad = " " * indent
    lines = []
    for i in range(0, len(x), per_line):
        chunk = x[i:i+per_line]
        lines.append(", ".join(str(v) for v in chunk))
    joined = (",\n" + pad + " ").join(lines)
    return f"[{joined}]"

ei_str = fmt_edge_index(src, dst, indent=len(f"    edge_index="))
x_str  = fmt_x(x, per_line=10, indent=len("    x="))

print(f"# {n_nodes} nodes, {n_edges} edges")
print(f"{VAR_NAME} = _make(")
print(f"    edge_index={ei_str},")
print(f"    x={x_str},")
print(f")")
