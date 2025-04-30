### DeepLoc_Reimplementation

A reimplementation of the original DeepLoc model in PyTorch.

### Run Instructions

To run the project first create an (anaconda) environment with torch (cuda support) and the various dependencies with

```bash
pip install -r requirements.txt
```

Then ensure the data resides in a data folder.

After this, use the following command (uses defaults).
python train.py -i data/train.npz -t data/test.npz

### Issues encountered

1. The original model uses Lasagne, which has poor support with non-Unix environment, rendering it nearly impossible for us to run.
   This is why we decided to re-implement the model ourselves in PyTorch

2. The dataset provided for any of the models shows discrepancies with the amount of data in mentioned in the paper
