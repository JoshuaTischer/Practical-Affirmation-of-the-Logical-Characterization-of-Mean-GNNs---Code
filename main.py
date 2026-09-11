import sys
import os
import csv
import random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from graphs.neighborhood_trees import nb_k00, nb_k01, nb_k02, nb_k03, nb_k04, nb_k05, nb_k06, nb_k07, nb_k08, nb_k09, nb_k10, nb_k11, nb_k12, nb_k13, nb_k14, nb_k15, nb_k16, nb_k17, nb_k18, nb_k19, nb_k20

from gnn import GNN
from label import y, atom, bot, top, neg, l_or, l_and, ml_box, ml_dm, gml_dm, rml_dm_geq, rml_dm_g, as_many_as


torch.set_printoptions(sci_mode=False)

GRAPH_FILES = [nb_k00, nb_k01, nb_k02, nb_k03, nb_k04, nb_k05, nb_k06, nb_k07, nb_k08, nb_k09, nb_k10, nb_k11, nb_k12, nb_k13, nb_k14, nb_k15, nb_k16, nb_k17, nb_k18, nb_k19, nb_k20]



BIAS = True
CLS_THRESHOLD = 0.5  # step function threshold for classification
ACTIVATION = "truncated_relu"  # options: relu, truncated_relu
SETTING = "uniform"  # options: uniform (train on degrees 0-9, test on degrees 10-20), non_uniform (train/test split evenly across all degrees)
EPOCHS = 200
EVAL_EVERY_EPOCHS = 40  # only run evaluate() every N epochs, always including the final epoch
LEARNING_RATE = 0.001
PRINT_EVERY_GRAPHS = 1000
MAX_TRIES = 3
LOSS_TOLERANCE = 0.9  # if loss does not decrease by this factor, restart training

def apply_labels(graphs, formula):
    for data in graphs:
        phi = formula(data.x)
        num_nodes = data.x.shape[0]
        data.y = y(data.edge_index, phi, num_nodes=num_nodes)  # label for every node, not just the root

    number_of_nodes = sum(data.y.shape[0] for data in graphs)
    number_of_true_labeled_nodes = sum(data.y.sum().item() for data in graphs)
    number_of_false_labeled_nodes = number_of_nodes - number_of_true_labeled_nodes
    print(f"Labeling: True - {number_of_true_labeled_nodes}/{number_of_nodes} ({number_of_true_labeled_nodes / number_of_nodes * 100:.2f}%), False - {number_of_false_labeled_nodes}/{number_of_nodes} ({number_of_false_labeled_nodes / number_of_nodes * 100:.2f}%)")
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

def group_nodes_by_degree(graphs, bin_size=5):
    """Buckets individual nodes (not whole graphs) by each node's own out-degree."""
    buckets = {}
    for data in graphs:
        num_nodes = data.x.shape[0]
        src = data.edge_index[0]
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        if src.numel() > 0:
            degrees.scatter_add_(0, src, torch.ones_like(src))
        bucket_low = (degrees // bin_size) * bin_size
        for low in bucket_low.unique().tolist():
            mask = bucket_low == low
            key = (low, low + bin_size - 1)
            entry = buckets.setdefault(key, {"total_nodes": 0, "positive_nodes": 0})
            entry["total_nodes"] += mask.sum().item()
            entry["positive_nodes"] += (data.y[mask] == 1).sum().item()
    return dict(sorted(buckets.items(), key=lambda item: item[0][0]))

def train_epoch(model, optimizer, train_graphs, epoch, total_graphs_processed=0, print_every_graphs=None):
    model.train()
    total_loss = 0.0
    for i, data in enumerate(train_graphs, start=1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[:, -1]  # all nodes, CLS reads last component
        loss = F.binary_cross_entropy_with_logits(out - 0.5, data.y.float()) # shift by -0.5 so the loss's implicit sigmoid boundary (0) lines up with CLS_THRESHOLD (0.5)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_graphs_processed += 1

        if print_every_graphs and total_graphs_processed % print_every_graphs == 0:
            running_loss = total_loss / i
            print(
                f"Epoch {epoch:03d} | Processed Graphs: {total_graphs_processed} | "
                f"Running Loss: {running_loss:.4f}"
            )

    return total_loss / len(train_graphs), total_graphs_processed

def evaluate(model, graphs):
    model.eval()
    total_correct = 0
    positive_correct = 0
    negative_correct = 0
    total_nodes = 0
    total_positive = 0
    total_negative = 0
    with torch.no_grad():
        for data in graphs:
            out = model(data.x, data.edge_index)[:, -1]  # all nodes, CLS reads last component
            prediction = (out >= CLS_THRESHOLD).long()          # step function
            total_correct += (prediction == data.y).sum().item()
            positive_correct += ((prediction == 1) & (data.y == 1)).sum().item()
            negative_correct += ((prediction == 0) & (data.y == 0)).sum().item()
            total_positive += (data.y == 1).sum().item()
            total_negative += (data.y == 0).sum().item()
            total_nodes += data.y.shape[0]
    return [total_correct, positive_correct, negative_correct, total_nodes, total_positive, total_negative]

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")
RESULTS_COLUMNS = [
    "aggregation", "formula", "dimension", "layers", "training_mask", "testing_mask",
    "total_nodes", "total_positive", "total_negative", "tries",
    "train_total_correct", "train_ratio_correct", "train_total_positive_correct", "train_ratio_positive_correct", "train_total_negative_correct", "train_ratio_negative_correct",
    "test_total_correct", "test_ratio_correct", "test_total_positive_correct", "test_ratio_positive_correct", "test_total_negative_correct", "test_ratio_negative_correct",
    "accuracies_by_neighbor_ranges_train", "accuracies_by_neighbor_ranges_test", "num_epochs", "progress"
]

def _ensure_csv_header():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=RESULTS_COLUMNS).writeheader()

def write_results_csv(row: dict):
    _ensure_csv_header()
    with open(RESULTS_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULTS_COLUMNS).writerow(row)

def one_cycle(training_mask, testing_mask, aggregation, dimension, layers, formula, formula_name, tries=1):

    print(f"Try {tries}")

    training_graphs = load_training_graphs(training_mask)
    random.shuffle(training_graphs)
    test_graphs = load_test_graphs(testing_mask)
    apply_labels(training_graphs + test_graphs, formula)

    train_nodes = sum(data.x.shape[0] for data in training_graphs)
    test_nodes = sum(data.x.shape[0] for data in test_graphs)
    print(f"Train: {train_nodes} nodes  |  Test: {test_nodes} nodes\n")

    # initialize GNN
    if layers < 1: 
        layers = 1
        dimension = 1
    input_dimension = (training_graphs or test_graphs)[0].x.shape[1] #get input dimension from first graph
    model = GNN([input_dimension] + [dimension] * layers, aggregation=aggregation, activation=ACTIVATION, bias=BIAS)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    last_loss = 0.0
    progress = ""
    total_graphs_processed = 0

    final_train_acc = None
    final_test_acc = None
    for epoch in range(1, EPOCHS + 1):
        loss, total_graphs_processed = train_epoch(
            model,
            optimizer,
            training_graphs,
            epoch=epoch,
            total_graphs_processed=total_graphs_processed,
            print_every_graphs=PRINT_EVERY_GRAPHS,
        )

        if epoch % EVAL_EVERY_EPOCHS != 0 and epoch != EPOCHS:
            continue

        train_results = evaluate(model, training_graphs)
        test_results = evaluate(model, test_graphs)

        if (train_results[0] == train_results[3]):
            final_train_acc = train_results
            final_test_acc = test_results
            print(f"\nPerfect accuracy achieved at epoch {epoch}. Stopping training.")
            print("\nFinal Evaluation")
        elif (epoch == EPOCHS):
            final_train_acc = train_results
            final_test_acc = test_results
            print("\nFinal Evaluation")
        else:
            print(f"\nEpoch {epoch:03d} Evaluation")

        string_epoch_results = ""
        string_epoch_results += f"Loss: {loss:.4f} \n"
        string_epoch_results += f"Train Acc: {train_results[0]}/{train_results[3]} ({train_results[0] / train_results[3] * 100:.2f}%) \n"
        string_epoch_results += f"Test Acc:  {test_results[0]}/{test_results[3]} ({test_results[0] / test_results[3] * 100:.2f}%) \n"
        string_epoch_results += f"Train Positive Acc: {train_results[1]}/{train_results[4]} ({train_results[1] / train_results[4] * 100:.2f}%) \n"
        string_epoch_results += f"Test Positive Acc:  {test_results[1]}/{test_results[4]} ({test_results[1] / test_results[4] * 100:.2f}%) \n"
        string_epoch_results += f"Train Negative Acc: {train_results[2]}/{train_results[5]} ({train_results[2] / train_results[5] * 100:.2f}%) \n"
        string_epoch_results += f"Test Negative Acc:  {test_results[2]}/{test_results[5]} ({test_results[2] / test_results[5] * 100:.2f}%) \n"
        progress += f"Epoch {epoch:03d}\n{string_epoch_results}\n"

        print(string_epoch_results)
        print("\n")


        if (train_results[0] == train_results[3]):
            break

        if last_loss > 0 and loss / last_loss > LOSS_TOLERANCE and tries < MAX_TRIES and train_results[0]/train_results[3] < 0.9:
            print(f"Loss did not decrease significantly last_loss={last_loss:.4f}, current_loss={loss:.4f} ({loss/last_loss:.4f} > {LOSS_TOLERANCE}). Restarting.")
            tries += 1
            one_cycle(training_mask, testing_mask, aggregation, dimension, layers, formula, formula_name, tries=tries)
            return
        last_loss = loss

    final_epoch = epoch
    final_try = tries

    neighbor_bin_size = 5
    bucket_ranges = [(start, start + neighbor_bin_size - 1) for start in range(0, 5 * neighbor_bin_size, neighbor_bin_size)]
    print(f"\n Accuracy by Root Neighbor Range (bin size = {neighbor_bin_size})")
    string_acc_by_ranges = ""
    string_acc_by_ranges_train = ""
    string_acc_by_ranges_test = ""
    fmt = lambda v: f"{v:.4f}" if v != 0.0000 else 0.0000

    def summarize_bucket(graphs):
        bucket_stats = {}
        for data in graphs:
            root_neighbors = (data.edge_index[0] == 0).sum().item()
            low = int(root_neighbors // neighbor_bin_size) * neighbor_bin_size
            high = low + neighbor_bin_size - 1
            bucket_stats.setdefault((low, high), []).append(data)

        summary = {}
        for (low, high), category_graphs in bucket_stats.items():
            total_correct = 0
            total_nodes = 0
            positive_correct = 0
            total_positive = 0
            negative_correct = 0
            total_negative = 0
            for data in category_graphs:
                out = model(data.x, data.edge_index)[:, -1]  # all nodes, CLS reads last component
                prediction = (out >= CLS_THRESHOLD).long()
                total_correct += (prediction == data.y).sum().item()
                positive_correct += ((prediction == 1) & (data.y == 1)).sum().item()
                negative_correct += ((prediction == 0) & (data.y == 0)).sum().item()
                total_positive += (data.y == 1).sum().item()
                total_negative += (data.y == 0).sum().item()
                total_nodes += data.y.shape[0]

            accuracy = total_correct / total_nodes
            accuracy_on_positives = positive_correct / total_positive if total_positive > 0 else 0.0000
            accuracy_on_negatives = negative_correct / total_negative if total_negative > 0 else 0.0000

            summary[(low, high)] = {
                "nodes": total_nodes,
                "accuracy": accuracy,
                "accuracy_on_positives": accuracy_on_positives,
                "accuracy_on_negatives": accuracy_on_negatives,
            }
        return summary

    model.eval()
    with torch.no_grad():
        train_bucket_stats = summarize_bucket(training_graphs)
        test_bucket_stats = summarize_bucket(test_graphs)

    def bucket_value(stats):
        if stats is None:
            return "n/a", 0
        value = f"{fmt(stats['accuracy'])} / {fmt(stats['accuracy_on_positives'])} / {fmt(stats['accuracy_on_negatives'])}"
        return value, stats["nodes"]

    for (low, high) in bucket_ranges:
        train_value, train_graph_count = bucket_value(train_bucket_stats.get((low, high)))
        test_value, test_graph_count = bucket_value(test_bucket_stats.get((low, high)))

        string_acc_by_ranges += (
            f"  [{low:3d}-{high:3d}] root neighbors | "
            f"train: nodes={train_graph_count:4d} acc={train_value:<24} | "
            f"test:  nodes={test_graph_count:4d} acc={test_value}\n"
        )
        string_acc_by_ranges_train += f"  [{low:3d}-{high:3d}] root neighbors | nodes={train_graph_count:4d} | accuracy={train_value}\n"
        string_acc_by_ranges_test += f"  [{low:3d}-{high:3d}] root neighbors | nodes={test_graph_count:4d} | accuracy={test_value}\n"

    print(string_acc_by_ranges)

    train_ratio_correct = round(final_train_acc[0] / final_train_acc[3], 4)
    train_ratio_positive_correct = round(final_train_acc[1] / final_train_acc[4], 4)
    train_ratio_negative_correct = round(final_train_acc[2] / final_train_acc[5], 4)
    test_ratio_correct = round(final_test_acc[0] / final_test_acc[3], 4)
    test_ratio_positive_correct = round(final_test_acc[1] / final_test_acc[4], 4)
    test_ratio_negative_correct = round(final_test_acc[2] / final_test_acc[5], 4)

    agg_label = aggregation.capitalize()
    train_acc_cell = f"$acc_{{tr}}={train_ratio_correct:.4f}$\\\\$acc_{{tr}}^{{+}}={train_ratio_positive_correct:.4f}$\\\\$acc_{{tr}}^{{-}}={train_ratio_negative_correct:.4f}$"
    test_acc_cell = f"$acc_{{ts}}={test_ratio_correct:.4f}$\\\\$acc_{{ts}}^{{+}}={test_ratio_positive_correct:.4f}$\\\\$acc_{{ts}}^{{-}}={test_ratio_negative_correct:.4f}$"

    print("\nLaTeX table row:")
    print(f"\\makecell{{{train_acc_cell}}} & % {agg_label}: Train Acc.")
    print(f"\\makecell{{{test_acc_cell}}} & % {agg_label}: Test Acc.")

    write_results_csv({
        "aggregation": aggregation,
        "formula": formula_name,
        "dimension": dimension,
        "layers": layers,
        "training_mask": training_mask,
        "testing_mask": testing_mask,
        "total_nodes": final_test_acc[3],
        "tries": final_try,
        "total_positive": final_test_acc[4],
        "total_negative": final_test_acc[5],
        "train_total_correct": final_train_acc[0],
        "train_ratio_correct": train_ratio_correct,
        "train_total_positive_correct": final_train_acc[1],
        "train_ratio_positive_correct": train_ratio_positive_correct,
        "train_total_negative_correct": final_train_acc[2],
        "train_ratio_negative_correct": train_ratio_negative_correct,
        "test_total_correct": final_test_acc[0],
        "test_ratio_correct": test_ratio_correct,
        "test_total_positive_correct": final_test_acc[1],
        "test_ratio_positive_correct": test_ratio_positive_correct,
        "test_total_negative_correct": final_test_acc[2],
        "test_ratio_negative_correct": test_ratio_negative_correct,
        "accuracies_by_neighbor_ranges_train": string_acc_by_ranges_train,
        "accuracies_by_neighbor_ranges_test": string_acc_by_ranges_test,
        "num_epochs": final_epoch,
        "progress": progress,
    })

def main():
    graphs_per_file = len(GRAPH_FILES[0].graphs)  # 886, same for every degree file
    nodes_per_graph = [module.graphs[0].x.shape[0] for module in GRAPH_FILES]
    node_budget = graphs_per_file * nodes_per_graph[0]
    graphs_per_degree = [min(graphs_per_file, node_budget // n) for n in nodes_per_graph]

    # non_uniform: every degree contributes to both train and test
    training_mask_non_uniform = [g // 2 for g in graphs_per_degree]
    testing_mask_non_uniform = [g - g // 2 for g in graphs_per_degree]

    # uniform: degrees 0-9 go entirely to train, degrees 10-20 entirely to test
    training_mask_uniform = [graphs_per_degree[k] if k < 10 else 0 for k in range(len(GRAPH_FILES))]
    testing_mask_uniform = [graphs_per_degree[k] if k >= 10 else 0 for k in range(len(GRAPH_FILES))]

    training_mask = training_mask_uniform if SETTING == "uniform" else training_mask_non_uniform
    testing_mask = testing_mask_uniform if SETTING == "uniform" else testing_mask_non_uniform

    fomulas = [
        # first Test
        lambda x: lambda edge_index, num_nodes: atom(x, 0),

        # AFML
        lambda x: lambda edge_index, num_nodes: ml_dm(edge_index, l_and(atom(x, 0), atom(x, 1)), num_nodes),

        # ML
        lambda x: lambda edge_index, num_nodes: l_and(ml_dm(edge_index, l_and(atom(x, 0), atom(x, 1)), num_nodes),ml_box(edge_index, l_or(atom(x, 0), atom(x, 1)), num_nodes)),

        # GML
        lambda x: lambda edge_index, num_nodes: gml_dm(edge_index, atom(x, 1), num_nodes, 4),
        lambda x: lambda edge_index, num_nodes: l_and(gml_dm(edge_index, atom(x, 0), num_nodes, 2), neg(gml_dm(edge_index, atom(x, 0), num_nodes, 7))),

        # RML
        lambda x: lambda edge_index, num_nodes: rml_dm_g(edge_index, atom(x, 0), num_nodes, 0.45),
        lambda x: lambda edge_index, num_nodes: l_or(rml_dm_geq(edge_index, atom(x, 1), num_nodes, 0.7), neg(rml_dm_g(edge_index, atom(x, 0), num_nodes, 0.3))),

        # nested RML
        lambda x: lambda edge_index, num_nodes: rml_dm_geq(edge_index, rml_dm_g(edge_index, atom(x, 0), num_nodes, 0.5), num_nodes, 0.8),
        lambda x: lambda edge_index, num_nodes: rml_dm_g(edge_index, rml_dm_geq(edge_index, rml_dm_g(edge_index, atom(x, 1), num_nodes,0.5), num_nodes, 0.4), num_nodes, 0.3),

        # non MSO
        lambda x: lambda edge_index, num_nodes: as_many_as(edge_index, atom(x, 0), atom(x, 1), num_nodes),
        lambda x: lambda edge_index, num_nodes: l_or(as_many_as(edge_index, atom(x, 0), atom(x, 1), num_nodes), as_many_as(edge_index, l_and(atom(x, 0),atom(x, 1)), neg(l_or(atom(x, 0), atom(x, 1))), num_nodes)),

        # Test subformulas
        lambda x: lambda edge_index, num_nodes: l_and(ml_dm(edge_index, l_and(neg(atom(x, 0)), atom(x, 1)), num_nodes),ml_box(edge_index, l_or(atom(x, 0), atom(x, 1)), num_nodes)),
        lambda x: lambda edge_index, num_nodes: neg(rml_dm_g(edge_index, atom(x, 0), num_nodes, 0.3)),
    ]

    

    formula_depths = [
        1,
        4,7,
        2,6,
        2,6,
        2,4,
        3,4,
        8,2
    ]
        
    formula_names = [
        "psi_0", 

        "psi_1",
        "psi_2",

        "psi_3",
        "psi_4",

        "psi_5",
        "psi_6",

        "psi_7",
        "psi_8",

        "psi_9",
        "psi_10",

        "test_1", "test_2",

    ]

    current_formula_indexes = [8]

    #one_cycle(training_mask=training_mask, testing_mask=testing_mask, aggregation="mean", dimension=formula_depths[current_formula_indexes[0]], layers=formula_depths[current_formula_indexes[0]], formula=fomulas[current_formula_indexes[0]], formula_name=formula_names[current_formula_indexes[0]])

    for current_formula_index in current_formula_indexes:
        print(f"Formula: {formula_names[current_formula_index]} | Depth: {formula_depths[current_formula_index]}")
        training_graphs = apply_labels(load_training_graphs(training_mask), fomulas[current_formula_index])
        test_graphs = apply_labels(load_test_graphs(testing_mask), fomulas[current_formula_index])


        def bucket_stats(stats):
            total_nodes = stats["total_nodes"]
            total_positive = stats["positive_nodes"]
            pct = total_positive / total_nodes * 100 if total_nodes else 0.0
            return total_nodes, total_positive, pct

        print(f"Positives on Neighbor numbers")
        train_degree_buckets = group_nodes_by_degree(training_graphs, bin_size=1)
        test_degree_buckets = group_nodes_by_degree(test_graphs, bin_size=1)
        empty_stats = {"total_nodes": 0, "positive_nodes": 0}
        for k in range(len(GRAPH_FILES)):
            low, high = k, k
            tr_n, tr_p, tr_pct = bucket_stats(train_degree_buckets.get((low, high), empty_stats))
            ts_n, ts_p, ts_pct = bucket_stats(test_degree_buckets.get((low, high), empty_stats))
            print(
                f"  [{low:3d}-{high:3d}] neighbors | "
                f"train: nodes={tr_n:4d} positives={tr_p:4d}/{tr_n:4d} ({tr_pct:6.2f}%) | "
                f"test: nodes={ts_n:4d} positives={ts_p:4d}/{ts_n:4d} ({ts_pct:6.2f}%)"
            )

        print(f"positive nodes on neighbor ranges (bin size = 5)")
        train_degree_buckets = group_nodes_by_degree(training_graphs, bin_size=5)
        test_degree_buckets = group_nodes_by_degree(test_graphs, bin_size=5)
        for low, high in [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24)]:
            tr_n, tr_p, tr_pct = bucket_stats(train_degree_buckets.get((low, high), empty_stats))
            ts_n, ts_p, ts_pct = bucket_stats(test_degree_buckets.get((low, high), empty_stats))
            print(
                f"  [{low:3d}-{high:3d}] neighbors | "
                f"train: nodes={tr_n:4d} positives={tr_p:4d}/{tr_n:4d} ({tr_pct:6.2f}%) | "
                f"test: nodes={ts_n:4d} positives={ts_p:4d}/{ts_n:4d} ({ts_pct:6.2f}%)"
            )

        def pos_pct(graphs_list):
            total_nodes = sum(data.y.shape[0] for data in graphs_list)
            positive = sum(data.y.sum().item() for data in graphs_list)
            return positive / total_nodes * 100 if total_nodes else 0.0

        all_graphs = training_graphs + test_graphs
        pos_all = pos_pct(all_graphs)
        pos_train = pos_pct(training_graphs)
        pos_test = pos_pct(test_graphs)
        pos_cell = "\\\\".join(f"{v:.2f}\\%" for v in (pos_all, pos_train, pos_test))

        print("\nLaTeX table row (Formula columns):")
        print(f"\\makecell{{{pos_cell}}} & % Pos. \\%")


if __name__ == "__main__":
    main()
