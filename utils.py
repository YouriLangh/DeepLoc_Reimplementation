import numpy as np
import torch
from tqdm import tqdm

class MetricTracker:
    def __init__(self, name):
        self.name = name
        self.reset()
        self.loss_total = 0.0
        self.correct = 0
        self.total = 0

    def reset(self):
        self.loss_total = 0.0
        self.correct = 0
        self.total = 0

    def update(self, loss, predictions, targets):
        self.loss_total += loss.item()
        self.correct += (predictions == targets).sum().item()
        self.total += targets.size(0)

    def average_loss(self, num_batches):
        return self.loss_total / num_batches if num_batches > 0 else 0.0

    def accuracy(self):
        return (self.correct / self.total) * 100 if self.total > 0 else 0.0



def run_epoch(model, dataloader, device, localization_criterion, membrane_criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loc_tracker = MetricTracker("Localization")
    mem_tracker = MetricTracker("Membrane")

    for batch in tqdm(dataloader, unit="batch") if is_train else dataloader:
        inputs, targets, masks, membrane_types = batch
        inputs, targets, masks, membrane_types = (
            inputs.to(device),
            targets.to(device),
            masks.to(device),
            membrane_types.to(device)
        )

        if is_train:
            optimizer.zero_grad()

        outputs, attention, context, membrane_out = model(inputs, masks)

        # Localization
        loss_loc = localization_criterion(outputs, targets)
        _, loc_pred = torch.max(outputs, 1)
        loc_tracker.update(loss_loc, loc_pred, targets)

        # Membrane
        loss_mem = membrane_criterion(membrane_out, membrane_types.float().unsqueeze(1))
        mem_pred = (torch.sigmoid(membrane_out) > 0.5).long()
        mem_tracker.update(loss_mem, mem_pred, membrane_types.long().unsqueeze(1))

        # Backward + optimize
        if is_train:
            total_loss = loss_loc + loss_mem
            total_loss.backward()
            optimizer.step()

    return loc_tracker, mem_tracker
