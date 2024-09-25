"""
[file]          cnn_1d.py
[description]   implement and evaluate WiFi-based model CNN-1D
"""
#
##
import time
import torch
import numpy as np
#
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from sklearn.metrics import classification_report, accuracy_score
#
from train_ssl import train_ssl
from preset import preset
# from classy_vision.generic.distributed_util import (
#     convert_to_distributed_tensor,
#     convert_to_normal_tensor,
#     is_distributed_training_run,
# )

#
##
## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- CNN-1D ------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
class CNN_1D(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(CNN_1D, self).__init__()
        #
        var_dim_input = var_x_shape[-1]
        # var_dim_output = var_y_shape[-1]
        #
        var_dim_output = 512
        self.layer_norm = torch.nn.BatchNorm1d(var_dim_input)
        #
        ##
        self.layer_cnn_1d_0 = torch.nn.Conv1d(in_channels=var_dim_input,
                                              out_channels=128,
                                              kernel_size=29,
                                              stride=13)
        #
        self.layer_cnn_1d_1 = torch.nn.Conv1d(in_channels=128,
                                              out_channels=256,
                                              kernel_size=15,
                                              stride=7)
        #
        self.layer_cnn_1d_2 = torch.nn.Conv1d(in_channels=256,
                                              out_channels=512,
                                              kernel_size=3,
                                              stride=1)
        #
        ##
        self.layer_linear = torch.nn.Linear(512, var_dim_output)
        #
        ##
        self.layer_dropout = torch.nn.Dropout(0.2)
        #
        self.layer_relu = torch.nn.ReLU()
        #
        torch.nn.init.xavier_uniform_(self.layer_cnn_1d_0.weight)
        torch.nn.init.xavier_uniform_(self.layer_cnn_1d_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_cnn_1d_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_linear.weight)

    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = torch.permute(var_t, (0, 2, 1))
        var_t = self.layer_norm(var_t)
        #
        var_t = self.layer_cnn_1d_0(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_cnn_1d_1(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = self.layer_cnn_1d_2(var_t)
        var_t = self.layer_relu(var_t)
        var_t = self.layer_dropout(var_t)

        var_t = torch.mean(var_t, dim=-1)

        var_t = self.layer_dropout(var_t)

        var_t = self.layer_linear(var_t)
        #
        var_output = var_t
        #
        return var_output

def infoNCE(nn, p, device="cpu", temperature=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    nn = gather_from_all(nn)
    p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).to(device=device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

class SS_Model(torch.nn.Module):
    def __init__(self,
                 var_x_shape,
                 var_y_shape):

        super().__init__()
        self.backbone = CNN_1D(var_x_shape, var_y_shape)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        sizes = [512, 256]
        # print(var_y_shape)
        self.online_head = torch.nn.Linear(sizes[0], var_y_shape[0])


        # Simclr projector sizes = [512, 256]
        layers = []

        # First layer
        layers.append(torch.nn.Linear(sizes[0], sizes[1], bias=False))
        layers.append(torch.nn.BatchNorm1d(sizes[1]))
        layers.append(torch.nn.ReLU(inplace=True))

        # Second layer (output layer)
        layers.append(torch.nn.Linear(sizes[1], sizes[1], bias=False))
        layers.append(torch.nn.BatchNorm1d(sizes[1]))

        self.projector = torch.nn.Sequential(*layers)

    def forward(self, y1, y2=None, labels=None, inference=False):
        if inference:
            r1 = self.backbone(y1)
            with torch.no_grad():
                logits = self.online_head(r1)
            return logits
        else:
            r1 = self.backbone(y1)
            r2 = self.backbone(y2)
            z1 = self.projector(r1)
            z2 = self.projector(r2)
            loss_ssl = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2
            logits = self.online_head(r1.detach())
            labels_flat = labels.reshape(-1,logits.shape[-1])
            if labels_flat.shape[-1] != logits.shape[-1]:
                raise ValueError(f"Mismatch in dimensions: labels_flat: {labels_flat.shape}, logits: {logits.shape}")

            loss_clc = self.bce_loss(logits, labels_flat.float())
            loss = loss_ssl + loss_clc
            return loss, logits

##


class TimeSeriesTransform:
    def __init__(self):
        self.transforms = [
            (self.add_jitter, 0.8),
            (self.scale_data, 0.7),
            (self.mask_random_segments, 0.6),
          #  (self.permute_segments, 0.5)
        ]

        self.transforms_prime = [
            (self.add_jitter, 0.9),
            (self.scale_data, 0.8),
            (self.mask_random_segments, 0.5),
            #(self.permute_segments, 0.4)
        ]

    def __call__(self, x):
        y1 = self.apply_transforms(x, self.transforms)
        y2 = self.apply_transforms(x, self.transforms_prime)
        return y1, y2

    def apply_transforms(self, x, transforms):
        for transform, p in transforms:
            if np.random.random() < p:
                x = transform(x)
        return x

    def add_jitter(self, data, noise_level=0.05):
        noise = torch.randn_like(data) * noise_level
        return data + noise

    def scale_data(self, data, scale_range=(0.9, 1.1)):
        scaling_factor = torch.FloatTensor(data.size(0), data.size(1)).uniform_(*scale_range)
        return data * scaling_factor

    def mask_random_segments(self, data, num_masks=1, mask_len=10):
        masked_data = data.clone()
        for sample in range(data.size(0)):
            for _ in range(num_masks):
                start = np.random.randint(0, data.size(-1) - mask_len)
                end = start + mask_len
                masked_data[start:end,:] = 0
        return masked_data
    # def mask_random_segments_along_channels(self, data, num_masks=1, mask_len=10):
    #     masked_data = data.clone()
    #     for sample in range(data.size(0)):
    #         for _ in range(num_masks):
    #             start = np.random.randint(0, data.size(-1) - mask_len)
    #             end = start + mask_len
    #             masked_data[start:end,:] = 0
    #     return masked_data
    # def mask_random_segments_along_antenna(self, data, num_masks=1, mask_len=10):
    #     masked_data = data.clone()
    #     for sample in range(data.size(0)):
    #         for _ in range(num_masks):
    #             start = np.random.randint(0, data.size(-1) - mask_len)
    #             end = start + mask_len
    #             masked_data[start:end,:] = 0
    #     return masked_data

    # def permute_segments(self, data, num_segments=3):
    #     total_length = data.size(-1)
    #     segment_length = total_length // num_segments
    #     remainder = total_length % num_segments
    #
    #     permuted_data = data.clone()
    #     for sample in range(data.size(0)):
    #         for channel in range(data.size(1)):
    #             segments = list(torch.split(data[sample, channel], segment_length))
    #             if remainder > 0:
    #                 segments[-1] = torch.cat([segments[-1], data[sample, channel, -remainder:]])
    #             np.random.shuffle(segments)
    #             permuted_data[sample, channel] = torch.cat(segments)
    #
    #     return permuted_data
    #

class CustomSSDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, transform=None):
        self.data_x = torch.from_numpy(data_x) if isinstance(data_x, np.ndarray) else data_x
        self.data_y = torch.from_numpy(data_y) if isinstance(data_y, np.ndarray) else data_y
        self.transform = transform if transform is not None else TimeSeriesTransform()

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]

        y1, y2 = self.transform(x)

        return y1, y2, y

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = torch.from_numpy(data_x) if isinstance(data_x, np.ndarray) else data_x
        self.data_y = torch.from_numpy(data_y) if isinstance(data_y, np.ndarray) else data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

def run_ssl(data_train_x,
            data_train_y,
            data_test_x,
            data_test_y,
            transform,
            var_repeat=10):
    """
    Run self-supervised WiFi-based model with two views

    Parameters:
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, labels to train model
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : transform: function, data augmentation transform
    : var_repeat: int, number of repeated experiments

    Returns:
    : result: dict, results of experiments
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)

    var_x_shape, var_y_shape = data_train_x[0].shape, data_train_y[0].reshape(-1).shape

    # Create custom datasets
    train_dataset = CustomSSDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    test_dataset = InferenceDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))
    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []

    # var_macs, var_params = get_model_complexity_info(SS_Model(var_x_shape, var_y_shape),
    #                                                  (var_x_shape, var_x_shape, var_y_shape), as_strings=False)
    #
    # print("Parameters:", var_params, "- FLOPs:", var_macs * 2)
    #
    for var_r in range(var_repeat):
        print("Repeat", var_r)

        torch.random.manual_seed(var_r + 39)

        model_ssl = torch.compile(SS_Model(var_x_shape, var_y_shape).to(device))

        optimizer = torch.optim.Adam(model_ssl.parameters(),
                                     lr=preset["nn"]["lr"],
                                     weight_decay=0)

        var_time_0 = time.time()

        # Train
        var_best_weight = train_ssl(model=model_ssl,
                                    optimizer=optimizer,
                                    data_train_set=train_dataset,
                                    data_test_set=test_dataset,
                                    var_threshold=preset["nn"]["threshold"],
                                    var_batch_size=preset["nn"]["batch_size"],
                                    var_epochs=preset["nn"]["epoch"],
                                    device=device)

        var_time_1 = time.time()

        # Test
        model_ssl.load_state_dict(var_best_weight)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=preset["nn"]["batch_size"], shuffle=False)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for y1, labels in test_loader:
                y1,  labels = y1.to(device), labels.to(device)
                predict_test_y = model_ssl(y1, inference=True)
                predict_test_y = (torch.sigmoid(predict_test_y) > preset["nn"]["threshold"]).float()
                all_preds.append(predict_test_y.cpu())
                all_labels.append(labels.cpu())

        predict_test_y = torch.cat(all_preds, dim=0).numpy()
        data_test_y = torch.cat(all_labels, dim=0).numpy()

        var_time_2 = time.time()

        # Evaluate
        data_test_y_c = data_test_y.reshape(-1, data_test_y.shape[-1])
        predict_test_y_c = predict_test_y.reshape(-1, data_test_y.shape[-1])

        result_acc = accuracy_score(data_test_y_c.astype(int),
                                    predict_test_y_c.astype(int))

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



def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if False:
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer.apply(tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor