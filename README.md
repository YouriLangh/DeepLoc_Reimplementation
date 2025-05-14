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

> Note: This is the training command, do not use.

To train using DeepLoc 1.0 dataset

```bash
python train.py -i data/deeploc1.0-train.csv -t data/deeploc1.0-test.csv -v DeepLoc1.0
```

To train using DeepLoc 2.1 dataset

```bash
python train.py -i data/deeploc2.1_training_processed.csv -t data/deeploc2.1_test_processed.csv -v DeepLoc2.0
```

```bash
python train.py -i data/deeploc2.1_training_processed.csv -t data/deeploc2.1_test_processed.csv --load_model results/DeepLoc2.0/final_model.pth --eval_only -v DeepLoc2.0
```

### Issues encountered

1. The original model uses Lasagne, which has poor support with non-Unix environment, rendering it nearly impossible for us to run.
   This is why we decided to re-implement the model ourselves in PyTorch

2. Using the DeepLoc 1.0 dataset required to ensure "stringent" homology partitioning. As PSI-CD-HIT was experienced to have bad non-Unix support, we used the 2.0/2.1 dataset which already does this partitioning for us. Nevertheless, this gives us ~7K more sequences than in DeepLoc 1.0, which should be considered when discussing accuracy. CD-HIT is a Unix developed tool, issues with BLAST prevented us from reimplementing this step.
