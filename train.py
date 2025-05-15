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
from utils import run_epoch, MetricTracker
import json 
import csv

label_columns = [
    "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion",
    "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"
]

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--trainset', help="npz file with training profiles data")
parser.add_argument('-t', '--testset', help="npz file with test profiles data to calculate final accuracy")
parser.add_argument('-bs', '--batch_size', help="Minibatch size, default = 128", default=128)
parser.add_argument('-e', '--epochs', help="Number of training epochs, default = 5", default=30) #200 normally
parser.add_argument('-n', '--n_filters', help="Number of filters, default = 10", default=10)
parser.add_argument('-lr', '--learning_rate', help="Learning rate, default = 0.0005", default=0.0005)
parser.add_argument('-id', '--in_dropout', help="Input dropout, default = 0.2", default=0.2)
parser.add_argument('-hd', '--hid_dropout', help="Hidden layers dropout, default = 0.5", default=0.5)
parser.add_argument('-hn', '--n_hid', help="Number of hidden units, default = 256", default=256)
parser.add_argument('-se', '--seed', help="Seed for random number init., default = 123456", default=123456)
parser.add_argument('--load_model', help="Path to saved model weights (.pth) to load", default=None)
parser.add_argument('--eval_only', action='store_true', help="Only evaluate the model, skip training")
parser.add_argument('-v', '--data_version', help="Which dataset version is used to evaluate the model?", default="DeepLoc1.0")

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

if args.load_model:
    print(f"Loading pretrained weights from {args.load_model}")
    model.load_state_dict(torch.load(args.load_model, map_location=device))


# optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=float(args.learning_rate),
    momentum=0.9,        
    weight_decay=1e-4    
)

# === Logging ===
train_val_logs = []


# === Training Loop ===
if not (args.eval_only or args.load_model):
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
        
        # # === Validation ===
        # val_loc_tracker, val_mem_tracker = run_epoch(
        #     model, test_loader, device,
        #     model.localization_criterion, model.membrane_criterion
        # )


        # print(f"Validation {val_loc_tracker.name} Loss: {val_loc_tracker.average_loss(len(test_loader)):.4f}, "
        #     f"{val_loc_tracker.name} Accuracy: {val_loc_tracker.accuracy():.2f}%, "
        #     f"{val_mem_tracker.name} Loss: {val_mem_tracker.average_loss(len(test_loader)):.4f}, "
        #     f"{val_mem_tracker.name} Accuracy: {val_mem_tracker.accuracy():.2f}%")
        print(f"Epoch time: {time.time() - start_time:.2f}s")
        train_val_logs.append({
            "epoch": epoch + 1,
            "loc_loss": loc_tracker.average_loss(len(train_loader)),
            "loc_acc": loc_tracker.accuracy(),
            "mem_loss": mem_tracker.average_loss(len(train_loader)),
            "mem_acc": mem_tracker.accuracy(),
            # "val_loc_loss": val_loc_tracker.average_loss(len(test_loader)),
            # "val_loc_acc": val_loc_tracker.accuracy(),
            # "val_mem_loss": val_mem_tracker.average_loss(len(test_loader)),
            # "val_mem_acc": val_mem_tracker.accuracy(),
        })

    pd.DataFrame(train_val_logs).to_csv("results/" + args.data_version + "train_val_metrics.csv", index=False)

# === Final Testing ===
model.eval()
mem_tracker = MetricTracker("Membrane")
loc_tracker = MetricTracker("Localization")

localization_conf = ConfusionMatrix(num_classes=10, class_names=label_columns.copy())
membrane_conf = ConfusionMatrix(num_classes=2, class_names=["Soluble", "Membrane-bound"])

test_preds = []
test_labels = []
membrane_preds = []


all_alphas = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        inputs, targets, masks, membrane_types = batch
        inputs, targets, masks, membrane_types = (
            inputs.to(device),
            targets.to(device),
            masks.to(device),
            membrane_types.to(device)
        )

        # Forward pass
        outputs, alphas, _, membrane_out = model(inputs, masks)
        alphas_1 = alphas[:, 1, :]  # Get the attention slice from decoding step 1 (as per the original code in their notebook)

        all_alphas.append(alphas_1.cpu())  # detach and move to CPU
        all_targets.append(targets.cpu())

        # === Localization ===
        _, loc_pred = torch.max(outputs, 1)
        loc_tracker.update(loss=torch.tensor(0.0), predictions=loc_pred, targets=targets)
        localization_conf.batch_add(targets.cpu().numpy(), loc_pred.cpu().numpy())
        test_preds.extend(loc_pred.cpu().numpy())
        test_labels.extend(targets.cpu().numpy())

        # === Membrane ===
        mem_pred = (torch.sigmoid(membrane_out) > 0.5).long()
        actual_mem = membrane_types.long().unsqueeze(1)
        mem_tracker.update(loss=torch.tensor(0.0), predictions=mem_pred, targets=actual_mem)
        membrane_conf.batch_add(actual_mem.cpu().numpy(), mem_pred.cpu().numpy())
        membrane_preds.extend(mem_pred.squeeze().cpu().numpy())

# === Print Results ===
print("Localization Confusion Matrix:")
print(localization_conf)
print(f"Localization Accuracy: {loc_tracker.accuracy():.2f}%")
print(f"Localization F1 per class: {localization_conf.F1()}")
print(f"Localization MCC per class: {localization_conf.matthews_correlation()}")
print(f"Sensitivity per class: {localization_conf.sensitivity()}")
print(f"Localization Overall MCC: {localization_conf.OMCC()}")
print(f"Gorodkin Score: {gorodkin(localization_conf.mat):.4f}")

print("\nMembrane Confusion Matrix:")
print(membrane_conf)
print(f"Membrane Accuracy: {mem_tracker.accuracy():.2f}%")
print(f"Membrane F1 per class: {membrane_conf.F1()}")
print(f"Membrane MCC per class: {membrane_conf.matthews_correlation()}")
print(f"Membrane Overall MCC: {membrane_conf.OMCC()}")

# === Save Model ===
if not (args.eval_only or args.load_model):
    torch.save(model.state_dict(), "results/" + args.data_version + "/final_model.pth")

# === Save Attention Weights ===
all_alphas = torch.cat(all_alphas, dim=0).numpy()    
all_targets = torch.cat(all_targets, dim=0).numpy()  

# Sort by label (like in DeepLoc 1.0)
sort_idx = np.argsort(all_targets)
sorted_alphas = all_alphas[sort_idx]
sorted_targets = all_targets[sort_idx]

# Save to .npy for reloading later
np.save("results/" + args.data_version + "/attention_step1_sorted.npy", sorted_alphas)
np.save("results/" + args.data_version + "/attention_labels_sorted.npy", sorted_targets)

# === Save Results ===
results_summary = {
    "localization": {
        "accuracy": loc_tracker.accuracy(),
        "gorodkin": gorodkin(localization_conf.mat)
    },
    "membrane": {
        "accuracy": mem_tracker.accuracy(),
        "overall_MCC": membrane_conf.OMCC()[0]
    }
}

with open("results/" + args.data_version + "/final_metrics.json", "w") as f:
    json.dump(results_summary, f, indent=4)


# === Save Localization Matrix with MCC and Sensitivity ===
loc_matrix = localization_conf.ret_mat()
mcc_per_class = localization_conf.matthews_correlation()
sensitivity_per_class = localization_conf.sensitivity()

loc_output_path = "results/" + args.data_version + "/localization_confusion_matrix.csv"
with open(loc_output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ["True\\Pred"] + label_columns + ["Total", "MCC", "Sensitivity"]
    writer.writerow(header)

    for i, row in enumerate(loc_matrix):
        total = sum(row)
        row_data = [label_columns[i]] + row.tolist() + [total, f"{mcc_per_class[i]:.4f}", f"{sensitivity_per_class[i]:.4f}"]
        writer.writerow(row_data)

    # Add column totals at the bottom
    col_totals = loc_matrix.sum(axis=0)
    writer.writerow(["Total"] + col_totals.tolist() + ["", "", ""])

print(f"Localization matrix saved to: {loc_output_path}")

# === Save Membrane Matrix ===
mem_matrix = membrane_conf.ret_mat()
mem_output_path = "results/" + args.data_version + "/membrane_confusion_matrix.csv"
with open(mem_output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["True\\Pred", "Soluble", "Membrane-bound", "Total"])
    for i, row in enumerate(mem_matrix):
        writer.writerow([["Soluble", "Membrane-bound"][i]] + row.tolist() + [sum(row)])
    writer.writerow(["Total"] + mem_matrix.sum(axis=0).tolist() + [""])

print(f"Membrane matrix saved to: {mem_output_path}")
