import time
import torch
import torch._dynamo
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import accuracy_score

torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 65536


def train_ssl(model: Module,
              optimizer: Optimizer,
              data_train_set,
              data_test_set,
              var_threshold: float,
              var_batch_size: int,
              var_epochs: int,
              device: device):
    """
    Generic training function for WiFi-based self-supervised models

    Parameters:
    : model: Pytorch model to train (SS_Model)
    : optimizer: optimizer to train model (e.g., Adam)
    : data_train_set: training set (CustomSSDataset)
    : data_test_set: test set (CustomSSDataset)
    : var_threshold: threshold to binarize sigmoid outputs
    : var_batch_size: batch size of each training step
    : var_epochs: number of epochs to train model
    : device: device (cuda or cpu) to train model

    Returns:
    : var_best_weight: weights of trained model
    """
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
            loss, logits = model(y1, y2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y1.size(0)
            total_samples += y1.size(0)

        avg_loss = total_loss / total_samples

        # Evaluate
        model.eval()
        with torch.no_grad():
            y1, y2, data_test_y = next(iter(data_test_loader))
            y1, y2, data_test_y = y1.to(device), y2.to(device), data_test_y.to(device)

            _, predict_test_y = model(y1, y2, data_test_y)

            var_loss_test = loss  # Note: You might need to calculate the test loss here if needed

            predict_test_y = (torch.sigmoid(predict_test_y) > var_threshold).float()

            data_test_y = data_test_y.detach().cpu().numpy()
            predict_test_y = predict_test_y.detach().cpu().numpy()

            # predict_test_y = predict_test_y # Assuming 30 is the correct final dimension
            # print(data_test_y.shape)
            data_test_y = data_test_y.reshape(-1, data_test_y.shape[-1])  # Reshape labels to match predictions

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

