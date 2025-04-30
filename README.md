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
