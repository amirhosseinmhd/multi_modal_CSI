import time
import torch
import numpy as np
#
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from sklearn.metrics import classification_report, accuracy_score
#
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
# from train import train
from preset import preset
from model.cnn_1d import CNN_1D


class DualBandCNN(torch.nn.Module):
    def __init__(self, var_x_shape_band1, var_x_shape_band2, var_y_shape):
        super(DualBandCNN, self).__init__()
        dimension_embedding = 512
        self.cnn_band1 = CNN_1D(var_x_shape_band1, [dimension_embedding])
        self.cnn_band2 = CNN_1D(var_x_shape_band2, [dimension_embedding])

        # Combine features
        self.combine_linear = torch.nn.Linear(dimension_embedding * 2, 512)  # 512 is the output size of each CNN_1D
        self.final_linear = torch.nn.Linear(512, var_y_shape[-1])

    def forward(self, x_band1, x_band2):
        features_band1 = self.cnn_band1(x_band1)
        features_band2 = self.cnn_band2(x_band2)

        combined_features = torch.cat((features_band1, features_band2), dim=1)
        combined_features = self.combine_linear(combined_features)
        output = self.final_linear(combined_features)
        return output


def run_dual_band(data_train_x_band1, data_train_y_band1,
                  data_test_x_band1, data_test_y_band1,
                  data_train_x_band2, data_train_y_band2,
                  data_test_x_band2, data_test_y_band2,
                  var_repeat=10):
    """
    [description]
    : run WiFi-based dual-band model
    [parameter]
    : data_train_x_band1, data_train_x_band2: numpy array, CSI amplitude to train model for each band
    : data_train_y_band1, data_train_y_band2: numpy array, labels to train model for each band
    : data_test_x_band1, data_test_x_band2: numpy array, CSI amplitude to test model for each band
    : data_test_y_band1, data_test_y_band2: numpy array, labels to test model for each band
    : var_repeat: int, number of repeated experiments
    [return]
    : result: dict, results of experiments
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess
    data_train_x_band1 = data_train_x_band1.reshape(data_train_x_band1.shape[0], data_train_x_band1.shape[1], -1)
    data_test_x_band1 = data_test_x_band1.reshape(data_test_x_band1.shape[0], data_test_x_band1.shape[1], -1)
    data_train_x_band2 = data_train_x_band2.reshape(data_train_x_band2.shape[0], data_train_x_band2.shape[1], -1)
    data_test_x_band2 = data_test_x_band2.reshape(data_test_x_band2.shape[0], data_test_x_band2.shape[1], -1)

    var_x_shape_band1, var_x_shape_band2 = data_train_x_band1[0].shape, data_train_x_band2[0].shape
    var_y_shape = data_train_y_band1[0].reshape(-1).shape

    data_train_set = TensorDataset(torch.from_numpy(data_train_x_band1),
                                   torch.from_numpy(data_train_x_band2),
                                   torch.from_numpy(data_train_y_band1))
    data_test_set = TensorDataset(torch.from_numpy(data_test_x_band1),
                                  torch.from_numpy(data_test_x_band2),
                                  torch.from_numpy(data_test_y_band1))

    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []

    # var_macs, var_params = get_model_complexity_info(DualBandCNN(var_x_shape_band1, var_x_shape_band2, var_y_shape),
    #                                                  (var_x_shape_band1, var_x_shape_band2), as_strings=False)
    #
    # print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    for var_r in range(var_repeat):
        print("Repeat", var_r)
        torch.random.manual_seed(var_r + 39)
        model_dual_band = torch.compile(DualBandCNN(var_x_shape_band1, var_x_shape_band2, var_y_shape).to(device))

        optimizer = torch.optim.Adam(model_dual_band.parameters(),
                                     lr=preset["nn"]["lr"],
                                     weight_decay=0)

        loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6] * var_y_shape[-1]).to(device))

        var_time_0 = time.time()

        var_best_weight = train(model=model_dual_band,
                                optimizer=optimizer,
                                loss=loss,
                                data_train_set=data_train_set,
                                data_test_set=data_test_set,
                                var_threshold=preset["nn"]["threshold"],
                                var_batch_size=preset["nn"]["batch_size"],
                                var_epochs=preset["nn"]["epoch"],
                                device=device)

        var_time_1 = time.time()

        # Test
        model_dual_band.load_state_dict(var_best_weight)
        model_dual_band.eval()

        with torch.no_grad():
            predict_test_y = model_dual_band(torch.from_numpy(data_test_x_band1).to(device),
                                             torch.from_numpy(data_test_x_band2).to(device))

        predict_test_y = (torch.sigmoid(predict_test_y) > preset["nn"]["threshold"]).float()
        predict_test_y = predict_test_y.detach().cpu().numpy()

        var_time_2 = time.time()

        # Evaluate
        data_test_y_c = data_test_y_band1.reshape(-1, data_test_y_band1.shape[-1])
        predict_test_y_c = predict_test_y.reshape(-1, data_test_y_band1.shape[-1])

        # Accuracy
        result_acc = accuracy_score(data_test_y_c.astype(int),
                                    predict_test_y_c.astype(int))

        # Report
        result_dict = classification_report(data_test_y_c,
                                            predict_test_y_c,
                                            digits=6,
                                            zero_division=0,
                                            output_dict=True)

        result["repeat_" + str(var_r)] = result_dict

        result_accuracy.append(result_acc)
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)

        print("repeat_" + str(var_r), result_accuracy)
        print(result)

    result["accuracy"] = {"avg": np.mean(result_accuracy), "std": np.std(result_accuracy)}
    result["time_train"] = {"avg": np.mean(result_time_train), "std": np.std(result_time_train)}
    result["time_test"] = {"avg": np.mean(result_time_test), "std": np.std(result_time_test)}
    # result["complexity"] = {"parameter": var_params, "flops": var_macs * 2}

    return result

def train(model, optimizer, loss, data_train_set, data_test_set, var_threshold, var_batch_size, var_epochs, device):
    saving_flag = False
    data_train_loader = DataLoader(data_train_set, var_batch_size, shuffle=True, pin_memory=True)
    data_test_loader = DataLoader(data_test_set, var_batch_size)
    var_best_accuracy = 0
    var_best_weight = None

    for var_epoch in range(var_epochs):
        model.train()
        var_time_e0 = time.time()

        for var_x_band1, var_x_band2, var_y in data_train_loader:
            var_x_band1, var_x_band2, var_y = var_x_band1.to(device), var_x_band2.to(device), var_y.to(device)
            optimizer.zero_grad()
            var_y_pred = model(var_x_band1, var_x_band2)

            var_loss_train = loss(var_y_pred, var_y.reshape(var_y.shape[0], -1 ).float())
            var_loss_train.backward()
            optimizer.step()

            # total_loss += loss.item() * y1.size(0)
            # total_samples += y1.size(0)

        # ... (validation code)
        # avg_loss = total_loss / total_samples

        # Evaluate
        model.eval()



        with torch.no_grad():
            var_x_band1, var_x_band2, var_y = next(iter(data_test_loader))
            var_x_band1, var_x_band2, var_y = var_x_band1.to(device), var_x_band2.to(device), var_y.to(device)
            var_y_pred = model(var_x_band1, var_x_band2)
            var_test_loss = loss(var_y_pred,  var_y.reshape(var_y.shape[0], -1 ).float() )# Check this


            # print("shape var y PRED", var_y_pred.shape)
            # print(" var y PRED", var_y.shape)
            # total_test_loss += var_test_loss.item() * var_x_band1.size(0)
            # total_test_samples += var_x_band1.size(0)

            var_y_pred = (torch.sigmoid(var_y_pred) > var_threshold).float()
            var_y = var_y.detach().cpu().numpy()
            var_y_pred = var_y_pred.detach().cpu().numpy()
            var_accuracy_test = accuracy_score(var_y.reshape(-1, var_y.shape[-1]).astype(int),
                                               var_y_pred.reshape(-1, var_y.shape[-1]).astype(int))
        print(f"Epoch {var_epoch}/{var_epochs}",
              "- %.6fs" % (time.time() - var_time_e0),
              "- Loss %.6f" % var_loss_train.cpu(),
              # "- Accuracy %.6f" % var_accuracy_train,
              "- Test Loss %.6f" % var_test_loss.cpu(),
              "- Test Accuracy %.6f" % var_accuracy_test)

        if var_accuracy_test > var_best_accuracy:
            #
            var_best_accuracy = var_accuracy_test
            var_best_weight = deepcopy(model.state_dict())
    return var_best_weight
