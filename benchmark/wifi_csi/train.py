import time
import torch
import torch._dynamo
import wandb
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from utils import calculate_matrix_absolute_error

torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 65536

def train(model: Module,
          optimizer: Optimizer,
          loss: Module,
          data_train_set: TensorDataset,
          data_test_set: TensorDataset,
          var_threshold: float,
          var_batch_size: int,
          var_epochs: int,
          device: device):

    """
    [description]
    : generic training function for WiFi-based models with wandb logging
    [parameter]
    : model: Pytorch model to train
    : optimizer: optimizer to train model (e.g., Adam)
    : loss: loss function to train model (e.g., BCEWithLogitsLoss)
    : data_train_set: training set
    : data_test_set: test set
    : var_threshold: threshold to binarize sigmoid outputs
    : var_batch_size: batch size of each training step
    : var_epochs: number of epochs to train model
    : device: device (cuda or cpu) to train model
    [return]
    : var_best_weight: weights of trained model
    """

    # Initialize wandb
    # wandb.init(project="wifi-based-model", config={
    #     "learning_rate": optimizer.param_groups[0]['lr'],
    #     "epochs": var_epochs,
    #     "batch_size": var_batch_size
    # })

    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    data_test_loader = DataLoader(data_test_set, len(data_test_set))

    var_loss_test_best = float('inf')
    var_best_weight = None

    for var_epoch in range(var_epochs):
        var_time_e0 = time.time()

        model.train()
        for data_batch in data_train_loader:
            data_batch_x, data_batch_y = data_batch
            data_batch_x = data_batch_x.to(device)
            data_batch_y = data_batch_y.to(device).sum(axis=1)

            predict_train_y = model(data_batch_x)

            var_loss_train = loss(predict_train_y, data_batch_y.float())

            optimizer.zero_grad()
            var_loss_train.backward()
            optimizer.step()

        # predict_train_y = (torch.sigmoid(predict_train_y) > var_threshold).float()

        data_batch_y = data_batch_y.detach().cpu().numpy()
        predict_train_y =  torch.clamp(torch.round(predict_train_y), min=0, max=5).float()
        predict_train_y = predict_train_y.detach().cpu().numpy()
        dict_error_train = calculate_matrix_absolute_error(data_batch_y.astype(int),
                                            predict_train_y.astype(int))

        model.eval()
        with torch.no_grad():
            data_test_x, data_test_y = next(iter(data_test_loader))
            data_test_x = data_test_x.to(device)
            data_test_y = data_test_y.to(device).sum(axis=1)

            predict_test_y = model(data_test_x)

            var_loss_test = loss(predict_test_y, data_test_y.float())

            predict_test_y = torch.clamp(torch.round(predict_test_y), min=0, max=5).float()

            data_test_y = data_test_y.detach().cpu().numpy().astype(int)
            predict_test_y = predict_test_y.detach().cpu().numpy().astype(int)
            dict_error_test = calculate_matrix_absolute_error(data_test_y, predict_test_y)

        # Log metrics to wandb
        wandb.log({
            "epoch": var_epoch,
            "train_loss": var_loss_train.item(),
            "test_loss": var_loss_test.item(),
            "total_error_train": dict_error_train['total_error'],
            "total_error_test": dict_error_test['total_error'],
            "perfect_prediction_percentage_test": dict_error_test['perfect_prediction_percentage'],
            "perfect_prediction_percentage_train": dict_error_train['perfect_prediction_percentage'],
            "accuracy_test": dict_error_test['accuracy'],
            "accuracy_train": dict_error_train['accuracy'],
        })

        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs"%(time.time() - var_time_e0),
              "- Loss %.6f"%var_loss_train.cpu(),
              "- Test Loss %.6f"%var_loss_test.cpu(),
              "- Total Error %.6f"%dict_error_test['total_error'],
              "- Perfect Prediction Percentage %.6f" % dict_error_test['perfect_prediction_percentage'],
              "- Accuracy Test%.6f" % dict_error_test['accuracy'],
               "Accuracy Train%.6f" %dict_error_train['accuracy'])

        if var_loss_test.cpu() < var_loss_test_best:
            var_loss_test_best = var_loss_test.cpu()
            var_best_weight = deepcopy(model.state_dict())

    # Finish the wandb run
    # wandb.finish()

    return var_best_weight