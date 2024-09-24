import time
import torch
import torch._dynamo
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 65536


def train_ssl(model: Module,
              optimizer: Optimizer,
              data_train_set: Dataset,
              data_test_set: Dataset,
              var_threshold: float,
              var_batch_size: int,
              var_epochs: int,
              device: device):
    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    data_test_loader = DataLoader(data_test_set, var_batch_size)

    var_best_accuracy = 0
    var_best_weight = None

    for var_epoch in range(var_epochs):
        # Train
        var_time_e0 = time.time()
        model.train()
        total_loss = 0
        total_samples = 0

        for y1, y2, labels in data_train_loader:
            y1, y2, labels = y1.to(device), y2.to(device), labels.to(device)
            optimizer.zero_grad()
            loss, _ = model(y1, y2, labels, False)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y1.size(0)
            total_samples += y1.size(0)

        avg_loss = total_loss / total_samples

        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, data_test_y in data_test_loader:
                x, data_test_y = x.to(device), data_test_y.to(device)
                predict_test_y = model(x, labels = data_test_y, inference=True)
                predict_test_y = (torch.sigmoid(predict_test_y) > var_threshold).float()
                all_preds.append(predict_test_y.cpu())
                all_labels.append(data_test_y.cpu())

        predict_test_y = torch.cat(all_preds, dim=0).numpy()
        data_test_y = torch.cat(all_labels, dim=0).numpy()

        data_test_y = data_test_y.reshape(-1, predict_test_y.shape[-1])
        var_accuracy_test = accuracy_score(data_test_y.astype(int), predict_test_y.astype(int))

        # Print
        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % avg_loss,
              "- Test Accuracy %.6f" % var_accuracy_test)

        if var_accuracy_test > var_best_accuracy:
            var_best_accuracy = var_accuracy_test
            var_best_weight = deepcopy(model.state_dict())

    if var_best_accuracy == 0:
        print("Warning: Accuracy did not improve during training. Returning final model state.")
        var_best_weight = model.state_dict()

    return var_best_weight
