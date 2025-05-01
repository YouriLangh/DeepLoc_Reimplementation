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

2. Using the DeepLoc 1.0 dataset required to ensure "stringent" homology partitioning. As we did not know how to do this, we used the 2.0/2.1 dataset which already does this partitioning for us. Nevertheless, this gives us ~7K more sequences than in DeepLoc 1.0, which should be considered when discussing accuracy.
