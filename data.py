import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import blosum as bl

matrix = bl.BLOSUM(62, default=0)
MAX_LEN = 1000  # Maximum length of the sequences

def encode_sequence_with_blosum(seq, matrix_keys, max_len=1000):
    """
    Encode a sequence using BLOSUM62, applying the mask.
    Returns: (max_len x 20) numpy array
    """
    encoded = np.zeros((max_len, len(matrix_keys)), dtype=np.float32)
    for i, aa in enumerate(seq):
        encoded[i] = [matrix[aa][other_aa] for other_aa in matrix_keys]
        # We ensured the default was 0, so no need to use the mask.
    return encoded


# Custom dataset class for DeepLoc
class DeepLocDataset(Dataset):
    def __init__(self, df, label_columns):
        self.sequences = df['PaddedSequence'].values # Fetch the processed sequences
        self.masks = df['Mask'].values # Fetch the masks
        self.labels = df[label_columns].values.astype(np.float32) # Convert labels to float32
        self.matrix = matrix # BLOSUM matrix
        self.max_len = MAX_LEN
        self.matrix_keys = list(matrix.keys())
        self.matrix_keys = [key for key in self.matrix_keys if key not in ['B', 'Z', 'X', 'J', '*']]
        self.num_features = len(self.matrix_keys)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        mask = self.masks[idx]
        label_vector = self.labels[idx]

        # Get index of class (assumes single label per sample)
        label = int(np.argmax(label_vector))
        encoded_seq = encode_sequence_with_blosum(seq, self.matrix_keys, self.max_len) # Encode the sequence using BLOSUM62
        mask_tensor = torch.tensor([int(m) for m in mask], dtype=torch.float32)
        return torch.tensor(encoded_seq), torch.tensor(label), mask_tensor # Return the encoded sequence, label, and mask tensor