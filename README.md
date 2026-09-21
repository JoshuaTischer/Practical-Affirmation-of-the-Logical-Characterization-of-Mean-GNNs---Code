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

Option 1: training mode

Uncomment the `one_cycle(...)` call in `main()` (or call `one_cycle` in a loop over aggregation, formula and size). Each call trains one model and appends one row to [results.csv](results.csv).

Option 2: formula infos

Uncomment the loop in 419 to label the graphs for the formulas listed in `current_formula_indexes` and print how many nodes are positive per neighborhood size and per range of 5. It also prints a LaTeX table row with the positive share (all / train / test). No model is trained.

Settings are constants at the top of [main.py](main.py):

| Constant | Default | Meaning |
|----------|---------|---------|
| `SETTING` | `uniform` | `uniform`: train on neighborhood sizes 0-9, test on 10-20. `non_uniform`: every size is split half/half into train and test |
| `ACTIVATION` | `truncated_relu` | `relu` or `truncated_relu` (clamped to [0, 1]) |
| `BIAS` | `True` | bias in the `C` matrix |
| `CLS_THRESHOLD` | `0.5` | step-function threshold on the last output component |
| `EPOCHS` | `200` | maximum number of epochs |
| `EVAL_EVERY_EPOCHS` | `40` | evaluation interval (the last epoch is always evaluated) |
| `LEARNING_RATE` | `0.001` | SGD learning rate |
| `PRINT_EVERY_GRAPHS` | `1000` | progress print interval during training |
| `MAX_TRIES` | `3` | maximum restarts of a run |
| `LOSS_TOLERANCE` | `0.9` | if the loss between two evaluations does not fall below this factor and train accuracy is below 90%, training restarts with fresh weights |

Training stops early once train accuracy reaches 100%.

## Method in brief

- **Model:** per layer `x <- act(C·x + A·AGG(neighbors))`. The classification is read from the last component of the final layer and thresholded at `CLS_THRESHOLD`. If the input feature dimension differs from the layer width, a linear input projection is added.
- **Data:** each graph is a complete-bipartite "neighborhood" graph K_{k,k} with 2 binary node features (4 feature types). File `nb_kNN` holds graphs with neighborhood size k = NN. Every distinct feature-type configuration of a layer appears at least once. Larger k means larger graphs, so the number of graphs used per size is scaled to a common node budget.
- **Labels:** computed exactly by evaluating the formula with [label.py](label.py) for every node.
- **Loss:** binary cross-entropy on the last component, shifted by -0.5 so the sigmoid boundary matches `CLS_THRESHOLD`.
- **Metrics:** total, positive-class and negative-class accuracy on train and test, accuracy per neighborhood-size range, and the training progress.

## Regenerating the datasets

```bash
python graphs/generate_neighborhood_trees.py
```

## Results

`results.csv` and `results_trees.csv` contain one row per training run (aggregation, formula, dimension, layers, masks, accuracies, per-size breakdown). Newly written rows follow `RESULTS_COLUMNS` in [main.py](main.py): setup (aggregation, formula, dimension, layers, masks), node counts, number of tries, train/test correct counts and ratios (overall, positive, negative), accuracies by neighborhood range, number of epochs and the training progress.