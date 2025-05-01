import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import DeepLocModel
from tqdm import tqdm
from confusionmatrix import ConfusionMatrix 
from metrics_mc import gorodkin
from data import DeepLocDataset
import pandas as pd
label_columns = [
    "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion",
    "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"
]

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--trainset', help="npz file with training profiles data")
parser.add_argument('-t', '--testset', help="npz file with test profiles data to calculate final accuracy")
parser.add_argument('-bs', '--batch_size', help="Minibatch size, default = 128", default=128)
parser.add_argument('-e', '--epochs', help="Number of training epochs, default = 5", default=5) #200 normally
parser.add_argument('-n', '--n_filters', help="Number of filters, default = 10", default=10)
parser.add_argument('-lr', '--learning_rate', help="Learning rate, default = 0.0005", default=0.0005)
parser.add_argument('-id', '--in_dropout', help="Input dropout, default = 0.2", default=0.2)
parser.add_argument('-hd', '--hid_dropout', help="Hidden layers dropout, default = 0.5", default=0.5)
parser.add_argument('-hn', '--n_hid', help="Number of hidden units, default = 256", default=256)
parser.add_argument('-se', '--seed', help="Seed for random number init., default = 123456", default=123456)
args = parser.parse_args()

# === Check for Data Files ===
if args.trainset is None:
    parser.print_help()
    exit(1)

# === Load the Data ===
train_df = pd.read_csv(args.trainset)
test_df = pd.read_csv(args.testset)

# === Create custom datasets
train_dataset = DeepLocDataset(train_df, label_columns)
test_dataset = DeepLocDataset(test_df, label_columns)

# === Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=True)

# === Set Random Seed ===
torch.manual_seed(int(args.seed))
np.random.seed(int(args.seed))

# === Initialize the Model ===
model = DeepLocModel(n_feat=20, n_class=10, n_hid=int(args.n_hid), n_filt=int(args.n_filters),
                     drop_per=float(args.in_dropout), drop_hid=float(args.hid_dropout))
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Move model to GPU if available

# === Define Loss Function and Optimizer ===
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

# === Training Loop ===
for epoch in range(int(args.epochs)):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    start_time = time.time()

    # Train on minibatches
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch"):
        inputs, targets, masks = batch
        inputs, targets, masks = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), \
                                  targets.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), \
                                  masks.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        optimizer.zero_grad()
        
        # Forward pass
        outputs, attention, context, membrane_out = model(inputs, masks)
        
        # Compute loss
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == targets).sum().item()
        total_preds += targets.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_preds / total_preds * 100

    # === Validation ===
    model.eval()
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, masks = batch
            inputs, targets, masks = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), \
                                      targets.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), \
                                      masks.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            outputs, attention, context, membrane_out = model(inputs, masks)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct_preds += (predicted == targets).sum().item()
            val_total_preds += targets.size(0)

    val_loss /= len(test_loader)
    val_acc = val_correct_preds / val_total_preds * 100

    # === Print Epoch Results ===
    print(f"Training loss: {epoch_loss:.4f}, Training accuracy: {epoch_acc:.2f}%")
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.2f}%")
    print(f"Epoch time: {time.time() - start_time:.3f}s")

# === Final Testing ===
model.eval()
test_preds = []
test_labels = []
membrane_preds = []

conf = ConfusionMatrix(num_classes=10, class_names=label_columns)

with torch.no_grad():
    for batch in test_loader:
        inputs, targets, masks = batch
        inputs, targets, masks = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), \
                                  targets.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), \
                                  masks.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Forward pass through the model
        outputs, _, _, membrane_out = model(inputs, masks)
        
        # Predict the class and collect predictions for evaluation
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(targets.cpu().numpy())
        
        # membrane_pred = (membrane_out > 0.5).float()

        # Update confusion matrix
        conf.batch_add(targets.cpu().numpy(), predicted.cpu().numpy())

# === Compute and Print Metrics for localization ===
# Compute confusion matrix and final accuracy
print("Confusion Matrix:")
print(conf)

# Calculate Accuracy
accuracy = conf.accuracy() * 100
print(f"Global Accuracy: {accuracy:.2f}%")

# Calculate F1 and MCC
f1 = conf.F1()
print(f"F1 per class: {f1}")

mcc = conf.matthews_correlation()
print(f"Matthews Correlation Coefficient per class: {mcc}")
print(f"Overall MCC: {conf.OMCC()}")

