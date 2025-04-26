import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import DeepLocModel  # This is the new PyTorch version we're building

# === Load the test data ===
data = np.load('data/test.npz') 
X_test = data['X_test']     # shape: (num_samples, seq_len, num_features)
y_test = data['y_test']     # shape: (num_samples,)
mask_test = data['mask_test']  # shape: (num_samples, seq_len)

print("Loaded test data")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("mask_test shape:", mask_test.shape)

# === Convert to PyTorch tensors ===
X_tensor = torch.tensor(X_test, dtype=torch.float32)
y_tensor = torch.tensor(y_test, dtype=torch.long)
mask_tensor = torch.tensor(mask_test, dtype=torch.float32)

# === Create DataLoader for batching ===
batch_size = 16
dataset = TensorDataset(X_tensor, y_tensor, mask_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size)

# === Initialize the model ===
model = DeepLocModel(n_feat=X_test.shape[2], n_class=10, n_hid=256, n_filt=10, drop_per=0.2, drop_hid=0.5)

model.eval()

# === Run a forward pass ===
with torch.no_grad():
    for batch in dataloader:
        X_batch, y_batch, mask_batch = batch
        out, attention, context = model(X_batch, mask_batch)

        print("Model output shape:", out.shape)           # expected: (batch_size, num_classes)
        print("Attention weights shape:", attention.shape)  # expected: (batch_size, decode_steps, seq_len)
        print("Context vector shape:", context.shape)     # expected: (batch_size, hidden_dim * 2)
        break