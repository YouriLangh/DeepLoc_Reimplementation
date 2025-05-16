# DeepLoc Reimplementation (PyTorch)

This project provides a PyTorch-based reimplementation of the [DeepLoc 1.0 model](https://services.healthtech.dtu.dk/services/DeepLoc-1.0/) for predicting protein subcellular localization from amino acid sequences.

More information can be found in our accompanying paper.

---

## Key Differences from the Original DeepLoc

The original DeepLoc 1.0 model was implemented using deprecated libraries (Theano, Lasagne) and relied on Unix-dependent tools (e.g., PSI-CD-HIT) for dataset partitioning. To address these limitations, we:

- Reimplemented the architecture in PyTorch 2.5.1 with CUDA 12.1 support.
- Replaced the hierarchical tree likelihood with a dense feedforward layer using softmax.
- PSI-CD-HIT is Unix-only; Used the fixed train-test split provided with the DeepLoc 1.0 dataset, avoiding cross-validation.
- Switched from SGD to the Adam optimizer for better convergence and stability.
- Limited input encoding to BLOSUM62 profiles due to computational constraints.
- Retained duplicate sequences to remain consistent with the original dataset distribution.
- We also extended the evaluation to the DeepLoc 2.0 dataset, treating the provided validation set as the test set after filtering for single-label sequences.

---

## Getting Started

### 1. Set up Environment

Ensure you have Python 3.10.16 (Anaconda recommended), PyTorch 2.5.1, and CUDA 12.1 installed. Then install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your data in the `./data` directory with the following names:

DeepLoc 1.0: `DeepLoc1.0/deeploc1.0.fasta`

DeepLoc 2.0: `DeepLoc2.0/deeploc_data_2.1.csv`

Run `data_explore.ipynb` to process and initialize the train/test sets for both datasets.

### 3. Training & Evaluation

The data comes with pre-trained weights for the model, we suggest you don't train the model yourself.
**DeepLoc 1.0 dataset:**

To train the model using the DeepLoc 1.0 dataset:

```bash
python main.py -i data/DeepLoc1.0/deeploc1.0-train.csv -t data/DeepLoc1.0/deeploc1.0-test.csv -v DeepLoc1.0
```

Evaluate with saved model for DeepLoc 1.0 dataset:

```bash
python main.py -i data/DeepLoc1.0/deeploc1.0-train.csv -t data/DeepLoc1.0/deeploc1.0-test.csv --load_model results/DeepLoc1.0/final_model.pth --eval_only -v DeepLoc1.0
```

**DeepLoc 2.0:**
To train the model using the DeepLoc 2.0 dataset:

```bash
python main.py -i data/DeepLoc2.0/deeploc2.1_training_processed.csv -t data/DeepLoc2.0/deeploc2.1_test_processed.csv -v DeepLoc2.0
```

Evaluate with saved model for DeepLoc 2.0 dataset:

```bash
python main.py -i data/DeepLoc2.0/deeploc2.1_training_processed.csv -t data/DeepLoc2.0/deeploc2.1_test_processed.csv --load_model results/DeepLoc2.0/final_model.pth --eval_only -v DeepLoc2.0
```

## Project Structure

`main.py`: Entry script for training/evaluation

`data_explore.ipynb`: Notebook initializing & exploring the input datasets.

`data/`: Input datasets

`results/`: Output results and saved models (organized per dataset)

`model.py`: DeepLoc model implementation

`data.py`: Data loading and encoding

`metrics_mc.py`: Metrics from the original authors

`confusionmatrix.py`: Confusion matrix logic from the original authors (patched to fix division by zero bugs)

`utils.py`: Training loop and metric tracker

`attention_visualization.ipynb`: Notebook to visualize attention weights per sequence in test set.

`results_visualization.ipynb`: Notebook to visualize the learning curve for the model.
