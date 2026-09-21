# Bachelor Thesis Code: Logical Expressiveness of GNNs

Code for the experiments in my bachelor thesis. It trains simple message-passing GNNs (with `max`, `sum` or `mean` aggregation) to learn node classifiers defined by modal-logic formulas, and measures how well they generalise from small to large neighborhoods.



## Repository layout

| Path | Content |
|------|---------|
| [main.py](main.py) | Experiment driver: builds train/test splits, trains the GNN, evaluates, appends a row to the results CSV |
| [gnn.py](gnn.py) | The GNN model (`C` and `A` matrices, configurable aggregation and activation) |
| [label.py](label.py) | Ground-truth labelling: evaluates modal formulas on a graph, node by node |
| [graphs/](graphs/) | Pre-generated datasets (`neighborhood_trees/nb_k00.py` … `nb_k20.py`) |
| [graphs/generate_neighborhood_trees.py](graphs/generate_neighborhood_trees.py) | Script that generates those datasets |
| [results.csv](results.csv), [results_trees.csv](results_trees.csv) | Raw results of the experiment runs |

## Setup

Python 3.10+ with:

```bash
pip install torch torch_geometric
```

There is no `requirements.txt`; only `torch` and `torch_geometric` (for `scatter` and `Data`) are needed.

## Running

```bash
python main.py
```

This sweeps over every combination of aggregation (`max`, `sum`, `mean`), formula, and model size, trains each model with SGD, and appends one row per run to the results CSV. Runs are long; the sweep is large.

Hyperparameters are constants at the top of [main.py](main.py):

| Constant | Default | Meaning |
|----------|---------|---------|
| `ACTIVATION` | `truncated_relu` | `relu` or `truncated_relu` (clamped to [0, 1]) |
| `BIAS` | `True` | bias in the `C` matrix |
| `CLS_THRESHOLD` | `0.5` | step-function threshold on the last output component |
| `EPOCHS` | see file | maximum number of epochs |
| `LEARNING_RATE` | `0.001` | SGD learning rate |

Model width and depth are derived from the formula's modal depth `d`: `dimension = layers = d + 2 + v` with variance `v ∈ {-2, …, 2}`.

## Method in brief

- **Model:** per layer `x ← act(C·x + A·AGG(neighbors))`. The classification is read from the last component of the final layer and thresholded at `CLS_THRESHOLD`. If the input feature dimension differs from the layer width, a linear input projection is added.
- **Data:** each graph is a complete-bipartite "neighborhood" graph K_{k,k} with 2 binary node features (4 feature types). File `nb_kNN` holds graphs with neighborhood size k = NN. Every distinct feature-type configuration of a layer appears at least once.
- **Labels:** computed exactly by evaluating the formula with [label.py](label.py), for every node.
- **Generalisation setting:** train and test masks select how many graphs per neighborhood size are used, so models can be trained on small neighborhoods and tested on larger ones (uniform setting) or trained and tested across all sizes (non-uniform setting).
- **Metrics:** train/test accuracy, plus accuracy split by neighborhood size, and the learned weight matrices.

## Regenerating the datasets

```bash
python graphs/generate_neighborhood_trees.py
```

## Results

`results.csv` and `results_trees.csv` contain one row per training run (aggregation, formula, dimension, layers, masks, accuracies, per-size breakdown). Note that the two files come from different versions of the driver and have slightly different column sets; the current `RESULTS_COLUMNS` in [main.py](main.py) defines the format of newly written rows.