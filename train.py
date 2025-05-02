import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import DeepLocModel
from tqdm import tqdm
from confusionmatrix import ConfusionMatrix 
from metrics_mc import gorodkin
from data import DeepLocDataset
import pandas as pd
from utils import run_epoch

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

# === Initialize Device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Define Loss Function and Optimizer ===
localization_criterion = torch.nn.CrossEntropyLoss()
criterion_membrane = nn.BCEWithLogitsLoss()

# === Initialize the Model ===
model = DeepLocModel(n_feat=20, n_class=10, n_hid=int(args.n_hid), n_filt=int(args.n_filters),
                     drop_per=float(args.in_dropout), drop_hid=float(args.hid_dropout), localization_criterion=localization_criterion,
                     membrane_criterion=criterion_membrane, learning_rate=float(args.learning_rate))
model.to(device)  # Move model to GPU if available

optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

# === Training Loop ===
for epoch in range(int(args.epochs)):
    start_time = time.time()

    # Training
    loc_tracker, mem_tracker = run_epoch(
        model, train_loader, device,
        model.localization_criterion, model.membrane_criterion, optimizer
    )
    # === Print Training Results ===
    print(f"Epoch {epoch + 1}/{args.epochs} - "
          f"{loc_tracker.name} Loss: {loc_tracker.average_loss(len(train_loader)):.4f}, "
          f"{loc_tracker.name} Accuracy: {loc_tracker.accuracy():.2f}%, "
          f"{mem_tracker.name} Loss: {mem_tracker.average_loss(len(train_loader)):.4f}, "
          f"{mem_tracker.name} Accuracy: {mem_tracker.accuracy():.2f}%")
    
    # === Validation ===
    val_loc_tracker, val_mem_tracker = run_epoch(
        model, test_loader, device,
        model.localization_criterion, model.membrane_criterion
    )

    print(f"Validation {val_loc_tracker.name} Loss: {val_loc_tracker.average_loss(len(test_loader)):.4f}, "
          f"{val_loc_tracker.name} Accuracy: {val_loc_tracker.accuracy():.2f}%, "
          f"{val_mem_tracker.name} Loss: {val_mem_tracker.average_loss(len(test_loader)):.4f}, "
          f"{val_mem_tracker.name} Accuracy: {val_mem_tracker.accuracy():.2f}%")
    print(f"Epoch time: {time.time() - start_time:.2f}s")

# === Final Testing ===
model.eval()
test_preds = []
test_labels = []
membrane_preds = []

conf = ConfusionMatrix(num_classes=10, class_names=label_columns)

with torch.no_grad():
    for batch in test_loader:
        inputs, targets, masks, membrane_types = batch
        inputs, targets, masks, membrane_types = inputs.to(device), \
                                      targets.to(device), \
                                      masks.to(device), \
                                      membrane_types.to(device)
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

