### DeepLoc_Reimplementation

A reimplementation of the original DeepLoc model in PyTorch.

### Run Instructions

1. First create an (anaconda) environment with torch (cuda support).

2. Then install the various dependencies using

```bash
pip install -r requirements.txt
```

3. Then ensure the data resides in the ./data location.

4. Running the code (uses defaults for the rest)
   python train.py -i data/train.npz -t data/test.npz

### Issues encountered

1. The original model uses Lasagne, which has poor support with non-Unix environment, rendering it nearly impossible for us to run.
   This is why we decided to re-implement the model ourselves in PyTorch

2. The dataset provided for any of the models shows discrepancies with the amount of data in mentioned in the paper. The gap in the size of the dataset from DeepLoc 1.0 is atleast 40% smaller than the one mentioned in their paper. For this reason, we decided to use the dataset from Deeploc 2.0 / 2.1.
