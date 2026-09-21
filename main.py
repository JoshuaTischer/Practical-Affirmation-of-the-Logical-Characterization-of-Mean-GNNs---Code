# TODO: this repo is linked publicly from the thesis (practical-affirmation.typ) but has no README --
# a few lines on how to run this script / label.py and regenerate results.csv would help readers and graders.
import sys
import os
import csv
import random
from xml.parsers.expat import model
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from graphs import rg_gs1_10, rg_gs11_20, rg_gs21_30, rg_gs31_40, rg_gs41_50, rg_gs51_60, rg_gs61_70, rg_gs71_80, rg_gs81_90, rg_gs91_100 # Random generated graphs
from graphs import cg_gs1_10, cg_gs11_20, cg_gs21_30, cg_gs31_40, cg_gs41_50, cg_gs51_60, cg_gs61_70, cg_gs71_80, cg_gs81_90, cg_gs91_100 # Claude generated graphs
from graphs import hc_gs1_10, hc_gs11_20, hc_gs21_30, hc_gs31_40, hc_gs41_50, hc_gs51_60, hc_gs61_70, hc_gs71_80, hc_gs81_90, hc_gs91_100 # Hand-crafted graphs

from gnn import GNN
from label import y, atom, bot, top, neg, l_or, l_and, ml_box, ml_dm, gml_dm, rml_dm_geq, rml_dm_g

from generate import RANGES

torch.set_printoptions(sci_mode=False)

RG_GRAPH_FILES = [rg_gs1_10, rg_gs11_20, rg_gs21_30, rg_gs31_40, rg_gs41_50, rg_gs51_60, rg_gs61_70, rg_gs71_80, rg_gs81_90, rg_gs91_100]
CG_GRAPH_FILES = [cg_gs1_10, cg_gs11_20, cg_gs21_30, cg_gs31_40, cg_gs41_50, cg_gs51_60, cg_gs61_70, cg_gs71_80, cg_gs81_90, cg_gs91_100]
HC_GRAPH_FILES = [hc_gs1_10, hc_gs11_20, hc_gs21_30, hc_gs31_40, hc_gs41_50, hc_gs51_60, hc_gs61_70, hc_gs71_80, hc_gs81_90, hc_gs91_100]
GRAPH_FILES = RG_GRAPH_FILES + CG_GRAPH_FILES + HC_GRAPH_FILES


""" def FORMULA(edge_index, input_dimension, data):
    return ml_dm(edge_index, ml_box(edge_index, atom(data.x, 2), data.num_nodes), data.num_nodes) """

BIAS = True
CLS_THRESHOLD = 0.5  # step function threshold for classification
ACTIVATION = "truncated_relu"  # options: relu, truncated_relu
EPOCHS = 2000
LEARNING_RATE = 0.001

def apply_labels(graphs, formula):
    for data in graphs:
        input_dimension = data.x.shape[1]
        data.y = formula(data.edge_index, input_dimension, data)
    return graphs


def load_training_graphs(training_mask):
    training_graphs = []
    for i, module in enumerate(GRAPH_FILES):
        training_graphs.extend(module.graphs[:training_mask[i]])
    return training_graphs

def load_test_graphs(testing_mask):
    test_graphs = []
    for i, module in enumerate(GRAPH_FILES):
        if testing_mask[i] > 0:
            test_graphs.extend(module.graphs[-testing_mask[i]:])
    return test_graphs

def load_test_graphs_by_set(graph_set):
    test_graphs = []
    for i, module in enumerate(graph_set):
        test_graphs.extend(module.graphs)
    return test_graphs
    


def train_epoch(model, optimizer, train_graphs):
    model.train()
    total_loss = 0.0
    for data in train_graphs:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[:, -1]  # CLS reads last component
        loss = F.binary_cross_entropy_with_logits(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_graphs)


def evaluate(model, graphs):
    model.eval()
    total_correct = 0
    total_nodes = 0
    with torch.no_grad():
        for data in graphs:
            out = model(data.x, data.edge_index)[:, -1]  # CLS reads last component
            prediction = (out >= CLS_THRESHOLD).long()          # step function
            total_correct += (prediction == data.y).sum().item()
            total_nodes += data.num_nodes
    return total_correct / total_nodes


RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")
RESULTS_COLUMNS = [
    "aggregation", "formula", "dimension", "layers",
    "training_mask", "testing_mask", "progress",
    "num_epochs", "final_train_acc", "final_test_acc",
    "accuracy_by_size_category", "accuracy_by_graph_set",
    "learned_C_matrix", "learned_A_matrix", "learned_bias_vector",
]

def _ensure_csv_header():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=RESULTS_COLUMNS).writeheader()

def write_results_csv(row: dict):
    _ensure_csv_header()
    with open(RESULTS_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULTS_COLUMNS).writerow(row)


def one_cycle(training_mask, testing_mask, aggregation, dimension, layers, formula, formula_name):

    training_graphs = apply_labels(load_training_graphs(training_mask), formula)
    test_graphs = apply_labels(load_test_graphs(testing_mask), formula)

    print(f"Train: {len(training_graphs)}  |  Test: {len(test_graphs)}\n")

    # initialize GNN
    input_dimension = (training_graphs or test_graphs)[0].x.shape[1] #get input dimension from first graph
    model = GNN([input_dimension] + [dimension] * layers, aggregation=aggregation, activation=ACTIVATION, bias=BIAS)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    best_test_accuracy = 0.0
    epochs_without_improvement = 0
    best_model_state = None
    progress = []  # list of dicts, one per printed epoch

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, optimizer, training_graphs)
        train_accuracy = evaluate(model, training_graphs)
        test_accuracy = evaluate(model, test_graphs)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            epochs_without_improvement = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0:
            category_accuracies = ""
            cat_acc_dict = {}
            with torch.no_grad():
                for low, high in RANGES:
                    category_graphs = [data for data in test_graphs
                               if low <= data.num_nodes <= high]
                    if not category_graphs:
                        continue
                    total_correct = sum(
                        ((model(data.x, data.edge_index)[:, -1] >= CLS_THRESHOLD).long() == data.y).sum().item()
                        for data in category_graphs
                    )
                    total_nodes = sum(data.num_nodes for data in category_graphs)
                    accuracy = total_correct / total_nodes
                    category_accuracies += f" {accuracy:.2f}"
                    cat_acc_dict[f"{low}-{high}"] = round(accuracy, 4)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f} | Category Acc:{category_accuracies}")
            progress.append({
                "epoch": epoch,
                "loss": round(loss, 6),
                "train_acc": round(train_accuracy, 4),
                "test_acc": round(test_accuracy, 4),
                "category_acc": cat_acc_dict,
            })

        if (train_accuracy == 1.0):
            print(f"\nPerfect accuracy achieved at epoch {epoch}. Stopping training.")
            break

    final_epoch = epoch
    model.load_state_dict(best_model_state)

    print("\nFinal Evaluation")
    final_train_acc = evaluate(model, training_graphs)
    final_test_acc = evaluate(model, test_graphs)
    print(f"Train Acc: {final_train_acc:.4f}")
    print(f"Test Acc:  {final_test_acc:.4f}")


    """ print("\nPer-graph Test Results")
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_graphs):
            out = model(data.x, data.edge_index)[:, -1]
            pred = (out >= CLS_THRESHOLD).long()
            acc = (pred == data.y).sum().item() / data.num_nodes
            print(f"  Graph {i}: pred={pred.tolist()}  true={data.y.tolist()}  acc={acc:.4f}")
     """

    print("\n Accuracy by Graph Size Category")
    acc_by_size = {}
    model.eval()
    with torch.no_grad():
        for low, high in RANGES:
            category_graphs = [data for data in test_graphs
                               if low <= data.num_nodes <= high]
            if not category_graphs:
                continue
            total_correct = sum(
                ((model(data.x, data.edge_index)[:, -1] >= CLS_THRESHOLD).long() == data.y).sum().item()
                for data in category_graphs
            )
            total_nodes = sum(data.num_nodes for data in category_graphs)
            accuracy = total_correct / total_nodes
            acc_by_size[f"{low}-{high}"] = round(accuracy, 4)
            print(f"  [{low:3d}-{high:3d}] nodes | graphs={len(category_graphs):3d} | accuracy={accuracy:.4f}")

    print("\nAccuracy on different sets of graphs:")
    rg_test_graphs = apply_labels(load_test_graphs_by_set(RG_GRAPH_FILES), formula)
    cg_test_graphs = apply_labels(load_test_graphs_by_set(CG_GRAPH_FILES), formula)
    hc_test_graphs = apply_labels(load_test_graphs_by_set(HC_GRAPH_FILES), formula)
    rg_acc = evaluate(model, rg_test_graphs)
    cg_acc = evaluate(model, cg_test_graphs)
    hc_acc = evaluate(model, hc_test_graphs)
    print(f"RG Acc:  {rg_acc:.4f}")
    print(f"CG Acc:  {cg_acc:.4f}")
    print(f"HC Acc:  {hc_acc:.4f}")
    acc_by_set = {"RG": round(rg_acc, 4), "CG": round(cg_acc, 4), "HC": round(hc_acc, 4)}

    # View the trained weight matrices
    print("\nTrained Weight Matrices")
    c_matrix = a_matrix = bias_vector = None
    for name, param in model.named_parameters():
        print(f"\n{name}:")
        print(f"Shape: {param.shape}")
        print(f"Values:\n{param.data}")
        if name == "C.weight":
            c_matrix = param.data.tolist()
        elif name == "A.weight":
            a_matrix = param.data.tolist()
        elif name == "C.bias":
            bias_vector = param.data.tolist()

    write_results_csv({
        "aggregation": aggregation,
        "formula": formula_name,
        "dimension": dimension,
        "layers": layers,
        "training_mask": training_mask,
        "testing_mask": testing_mask,
        "progress": progress,
        "num_epochs": final_epoch,
        "final_train_acc": round(final_train_acc, 4),
        "final_test_acc": round(final_test_acc, 4),
        "accuracy_by_size_category": acc_by_size,
        "accuracy_by_graph_set": acc_by_set,
        "learned_C_matrix": c_matrix,
        "learned_A_matrix": a_matrix,
        "learned_bias_vector": bias_vector,
    })

def main():
    training_masks = [
        [5]*10 + [0]*10 + [0]*10,
        [0]*10 + [5]*10 + [0]*10,
        [0]*10 + [0]*10 + [5]*10,
        [5]*10 + [5]*10 + [5]*10
    ]

    testing_masks = [
        [5]*10 + [0]*10 + [0]*10,
        [0]*10 + [5]*10 + [0]*10,
        [0]*10 + [0]*10 + [5]*10,
        [5]*10 + [5]*10 + [5]*10
    ]

    aggregations = ["max", "sum", "mean"]

    fomulas = [
        # AFML
        lambda edge_index, input_dimension, data: atom(data.x, 0),
        lambda edge_index, input_dimension, data: ml_box(edge_index, atom(data.x, 1), data.num_nodes),
        lambda edge_index, input_dimension, data: ml_dm(edge_index, l_and(ml_box(edge_index, bot(data.x), data.num_nodes), atom(data.x, 2)), data.num_nodes),
        lambda edge_index, input_dimension, data: ml_box(edge_index, l_or(atom(data.x, 0), l_or(ml_dm(edge_index, atom(data.x, 1), data.num_nodes), ml_box(edge_index, neg(atom(data.x, 2)), data.num_nodes))), data.num_nodes),

        # ML
        lambda edge_index, input_dimension, data: l_and(ml_dm(edge_index, top(data.x), data.num_nodes), ml_box(edge_index, ml_box(edge_index, bot(data.x), data.num_nodes), data.num_nodes)),
        lambda edge_index, input_dimension, data: ml_dm(edge_index, l_and(atom(data.x, 0), l_and(atom(data.x, 1), ml_box(edge_index, atom(data.x, 2), data.num_nodes))), data.num_nodes),
        lambda edge_index, input_dimension, data: ml_dm(edge_index, ml_dm(edge_index, ml_dm(edge_index, ml_dm(edge_index, ml_dm(edge_index, ml_box(edge_index, atom(data.x, 1), data.num_nodes), data.num_nodes), data.num_nodes), data.num_nodes), data.num_nodes), data.num_nodes),
        lambda edge_index, input_dimension, data: ml_dm(edge_index, ml_box(edge_index, ml_dm(edge_index, ml_box(edge_index, ml_dm(edge_index, ml_box(edge_index, bot(data.x), data.num_nodes), data.num_nodes), data.num_nodes), data.num_nodes), data.num_nodes), data.num_nodes),

        # GML
        lambda edge_index, input_dimension, data: gml_dm(edge_index, atom(data.x, 1), data.num_nodes, 3),
        lambda edge_index, input_dimension, data: l_and(gml_dm(edge_index, atom(data.x, 1), data.num_nodes, 3),neg(gml_dm(edge_index, atom(data.x, 1), data.num_nodes, 4))),
        lambda edge_index, input_dimension, data: gml_dm(edge_index, gml_dm(edge_index, top(data.x), data.num_nodes, 3), data.num_nodes, 1),
        lambda edge_index, input_dimension, data: l_or(gml_dm(edge_index, atom(data.x, 0), data.num_nodes, 2), l_or(gml_dm(edge_index, atom(data.x, 1), data.num_nodes, 2), gml_dm(edge_index, atom(data.x, 2), data.num_nodes, 2))),

        # RML
        lambda edge_index, input_dimension, data: rml_dm_geq(edge_index, atom(data.x, 1), data.num_nodes, 0.5),
        lambda edge_index, input_dimension, data: rml_dm_g(edge_index, l_and(neg(atom(data.x, 1)), neg(atom(data.x, 2))), data.num_nodes, 0.9),
        lambda edge_index, input_dimension, data: l_and(rml_dm_geq(edge_index, neg(atom(data.x, 0)), data.num_nodes, 0.1), l_and(rml_dm_geq(edge_index, neg(atom(data.x, 1)), data.num_nodes, 0.1), rml_dm_geq(edge_index, neg(atom(data.x, 2)), data.num_nodes, 0.1))),
        lambda edge_index, input_dimension, data: ml_box(edge_index, rml_dm_g(edge_index, atom(data.x, 0), data.num_nodes, 0.7), data.num_nodes),

    
    ]

    depths = [0, 1, 2, 2,
              2, 2, 6, 6, 
              1, 1, 2, 1,
              1, 1, 3, 2]
    
    formula_names = [
        "psi_1", "psi_2", "psi_3", "psi_4",
        "psi_5", "psi_6", "psi_7", "psi_8",
        "psi_9", "psi_10", "psi_11", "psi_12",
        "psi_13", "psi_14", "psi_15", "psi_16"
    ]

    """ for formula, depth in zip(fomulas, depths):
        test = apply_labels(load_test_graphs_by_set(RG_GRAPH_FILES), formula)
        print(f"Formula: {formula.__name__}, Number of true labeled nodes / all nodes: {sum(data.y.sum().item() for data in test)} / {sum(data.num_nodes for data in test)}") """
    

    for training_mask, testing_mask in zip(training_masks, testing_masks):
        for aggregation in aggregations:
            for formula, depth, formula_name in zip(fomulas, depths, formula_names):
                for variance in [-2, -1, 0, 1, 2]:
                    dimension = depth + 2 + variance
                    if dimension < 2:
                        continue    
                    print(f"\nTraining Mask: {training_mask}")
                    print(f"Testing Mask: {testing_mask}")
                    print(f"Aggregation: {aggregation}")
                    print(f"Formula: {formula_name} (depth={depth})")
                    one_cycle(training_mask, testing_mask, aggregation, dimension=dimension, layers=dimension, formula=formula, formula_name=formula_name)

if __name__ == "__main__":
    main()
