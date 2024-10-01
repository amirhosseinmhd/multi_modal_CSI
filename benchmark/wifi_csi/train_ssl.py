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
              device: device,
              save_path: str = "best_model.pth"):
    saving_flag = False
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
        # all_preds = []
        # all_labels = []

        with torch.no_grad():
            x, data_test_y = next(iter(data_test_loader))# x.to(device), data_test_y.to(device)
            x = x.to(device)
            data_test_y = data_test_y.to(device)

            predict_test_y = model(x, inference=True)
            predict_test_y = (torch.sigmoid(predict_test_y) > var_threshold).float()
            predict_test_y = predict_test_y.detach().cpu().numpy()
            #
            predict_test_y = predict_test_y.reshape(-1, data_test_y.shape[-1])
            data_test_y = data_test_y.reshape(-1, data_test_y.shape[-1]).cpu().numpy()
            #
            var_accuracy_test = accuracy_score(data_test_y.astype(int),
                                               predict_test_y.astype(int))

        # Print
        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % avg_loss,
              "- Test Accuracy %.6f" % var_accuracy_test)

        if saving_flag and var_accuracy_test > var_best_accuracy:
            var_best_accuracy = var_accuracy_test
            var_best_weight = deepcopy(model.state_dict())

            # Save the best model
            torch.save({
                'epoch': var_epoch,
                'model_state_dict': var_best_weight,
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': var_best_accuracy,
            }, save_path)
            print(f"Saved best model with accuracy {var_best_accuracy:.6f} to {save_path}")

    if var_best_accuracy == 0:
        print("Warning: Accuracy did not improve during training. Saving final model state.")
        var_best_weight = model.state_dict()
    return var_best_weight