import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import DeepLocModel
from tqdm import tqdm
from confusionmatrix import ConfusionMatrix 


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
if args.trainset is None or args.testset is None:
    parser.print_help()
    exit(1)

# === Load the Data ===
train_data = np.load(args.trainset)
test_data = np.load(args.testset)

# As sequences can be variable length, the dataset has to pad certain sequences. But our attention mechanism/ LSTM cell should not learn from padding
# Mask ensures we know which positions are padding and which are not (1 = not padding, 0 = padding)
X_train, y_train, mask_train = train_data['X_train'], train_data['y_train'], train_data['mask_train']

X_test, y_test, mask_test = test_data['X_test'], test_data['y_test'], test_data['mask_test']

# === Convert to PyTorch Tensors ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
mask_train_tensor = torch.tensor(mask_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
mask_test_tensor = torch.tensor(mask_test, dtype=torch.float32)

# === Create DataLoader for Batching ===
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, mask_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor, mask_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False)

# === Initialize the Model ===
model = DeepLocModel(n_feat=X_train.shape[2], n_class=10, n_hid=int(args.n_hid), n_filt=int(args.n_filters),
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
        outputs, attention, context = model(inputs, masks)
        
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
class_names = ['Nucleus', 'Cytoplasm', 'Extracellular', 'Mitochondrion', 'Cell membrane','Endoplasmic reticulum' ,'Plastid', 'Golgi apparatus', 'Lysosome/Vacuole', 'Peroxisome' ]
conf = ConfusionMatrix(num_classes=10, class_names=class_names)

with torch.no_grad():
    for batch in test_loader:
        inputs, targets, masks = batch
        inputs, targets, masks = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), \
                                  targets.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), \
                                  masks.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        outputs, _, _ = model(inputs, masks)
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(targets.cpu().numpy())
        conf.batch_add(
    targets.cpu().numpy(),
    predicted.cpu().numpy()
)


# Compute confusion matrix and final accuracy
print("Confusion Matrix:")
print(conf)

print(f"Global Accuracy: {conf.accuracy() * 100:.2f}%")
print(f"F1 per class: {conf.F1()}")
print(f"Matthews Correlation Coefficient per class: {conf.matthews_correlation()}")
print(f"Overall MCC: {conf.OMCC()}")

