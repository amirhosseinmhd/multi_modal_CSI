"""
[file]          detr.py
[description]   implement and evaluate WiFi-based model THAT_ENCODER
                https://github.com/windofshadow/THAT
"""
#
##
import os
import time
import torch
import numpy as np
from sklearn.model_selection import train_test_split
#
import torch.nn as nn
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from itertools import permutations
from sklearn.metrics import classification_report, accuracy_score
from scipy.optimize import linear_sum_assignment
from train import train
from preset import preset
import torch.nn.functional as F
from utils import *
import wandb


#
##
## ------------------------------------------------------------------------------------------ ##
## ----------------------------------- Gaussian Encoding ------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
class Gaussian_Position(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_dim_feature,
                 var_dim_time,
                 var_num_gaussian=10):
        #
        ##
        super(Gaussian_Position, self).__init__()
        #
        ## var_embedding: shape (var_dim_k, var_dim_feature)
        var_embedding = torch.zeros([var_num_gaussian, var_dim_feature], dtype=torch.float)
        self.var_embedding = torch.nn.Parameter(var_embedding, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.var_embedding)
        #
        ## var_position: shape (var_dim_time, var_dim_k)
        var_position = torch.arange(0.0, var_dim_time).unsqueeze(1).repeat(1, var_num_gaussian)
        self.var_position = torch.nn.Parameter(var_position, requires_grad=False)
        #
        ## var_mu: shape (1, var_dim_k)
        var_mu = torch.arange(0.0, var_dim_time, var_dim_time / var_num_gaussian).unsqueeze(0)
        self.var_mu = torch.nn.Parameter(var_mu, requires_grad=True)
        #
        ## var_sigma: shape (1, var_dim_k)
        var_sigma = torch.tensor([50.0] * var_num_gaussian).unsqueeze(0)
        self.var_sigma = torch.nn.Parameter(var_sigma, requires_grad=True)

    #
    ##
    def calculate_pdf(self,
                      var_position,
                      var_mu,
                      var_sigma):
        #
        ##
        var_pdf = var_position - var_mu  # (position-mu)
        #
        var_pdf = - var_pdf * var_pdf  # -(position-mu)^2
        #
        var_pdf = var_pdf / var_sigma / var_sigma / 2  # -(position-mu)^2 / (2*sigma^2)
        #
        var_pdf = var_pdf - torch.log(var_sigma)  # -(position-mu)^2 / (2*sigma^2) - log(sigma)
        #
        return var_pdf

    #
    ##
    def forward(self,
                var_input):
        var_pdf = self.calculate_pdf(self.var_position, self.var_mu, self.var_sigma)

        var_pdf = torch.softmax(var_pdf, dim=-1)
        #
        var_position_encoding = torch.matmul(var_pdf, self.var_embedding)
        #
        # print(var_input.shape, var_position_encoding.shape)
        var_output = var_input + var_position_encoding.unsqueeze(0)
        #
        return var_output


#
##
## ------------------------------------------------------------------------------------------ ##
## --------------------------------------- Encoder ------------------------------------------ ##
## ------------------------------------------------------------------------------------------ ##
#
##
class Encoder(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_dim_feature,
                 var_num_head=10,
                 var_size_cnn=[1, 3, 5]):
        #
        ##
        super(Encoder, self).__init__()
        #
        ##
        self.layer_norm_0 = torch.nn.LayerNorm(var_dim_feature, eps=1e-6)
        self.layer_attention = torch.nn.MultiheadAttention(var_dim_feature,
                                                           var_num_head,
                                                           batch_first=True)
        #
        self.layer_dropout_0 = torch.nn.Dropout(0.1)
        #
        ##
        self.layer_norm_1 = torch.nn.LayerNorm(var_dim_feature, 1e-6)
        #
        layer_cnn = []
        #
        for var_size in var_size_cnn:
            #
            layer = torch.nn.Sequential(torch.nn.Conv1d(var_dim_feature,
                                                        var_dim_feature,
                                                        var_size,
                                                        padding="same"),
                                        torch.nn.BatchNorm1d(var_dim_feature),
                                        torch.nn.Dropout(0.1),
                                        torch.nn.LeakyReLU())
            layer_cnn.append(layer)
        #
        self.layer_cnn = torch.nn.ModuleList(layer_cnn)
        #
        self.layer_dropout_1 = torch.nn.Dropout(0.1)

    #
    ##
    def forward(self,
                var_input):
        #
        ##
        var_t = var_input
        #
        var_t = self.layer_norm_0(var_t)
        #
        var_t, _ = self.layer_attention(var_t, var_t, var_t)

        var_t = self.layer_dropout_0(var_t)
        #
        var_t = var_t + var_input
        #
        ##
        var_s = self.layer_norm_1(var_t)

        var_s = torch.permute(var_s, (0, 2, 1))
        #
        var_c = torch.stack([layer(var_s) for layer in self.layer_cnn], dim=0)
        #
        var_s = torch.sum(var_c, dim=0) / len(self.layer_cnn)
        #
        var_s = self.layer_dropout_1(var_s)

        var_s = torch.permute(var_s, (0, 2, 1))
        #
        var_output = var_s + var_t
        #
        return var_output


#
##
## ------------------------------------------------------------------------------------------ ##
## ---------------------------------------- THAT_ENCODER -------------------------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
##
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=270):
        super().__init__()

        # Left stream - target 100 tokens
        self.left_extractor = nn.Sequential(
            # Initial noise reduction layer
            nn.Conv1d(input_channels, input_channels, kernel_size=5, padding=2, groups=input_channels),
            nn.BatchNorm1d(input_channels),
            nn.LeakyReLU(),

            # First denoising block
            nn.Conv1d(input_channels, input_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_channels * 2),
            nn.LeakyReLU(),
            nn.Conv1d(input_channels * 2, input_channels, kernel_size=3, padding=1, groups=input_channels),
            nn.BatchNorm1d(input_channels),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),  # Reduce time dimension by 2

            # Second denoising block with residual connection
            ResidualDenoising(input_channels),
            nn.AvgPool1d(kernel_size=2, stride=2),  # Further reduce time dimension

            # Final adjustment to get target sequence length
            nn.Conv1d(input_channels, input_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(input_channels),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(100)  # Force output to 100 tokens
        )

        # Right stream - target 50 tokens
        self.right_extractor = nn.Sequential(
            # Initial noise reduction
            nn.Conv1d(input_channels, input_channels, kernel_size=7, padding=3, groups=input_channels),
            nn.BatchNorm1d(input_channels),
            nn.LeakyReLU(),

            # Deeper denoising block
            nn.Conv1d(input_channels, input_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(input_channels * 2),
            nn.LeakyReLU(),
            nn.Conv1d(input_channels * 2, input_channels, kernel_size=5, padding=2, groups=input_channels),
            nn.BatchNorm1d(input_channels),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=4, stride=4),  # Aggressive early reduction

            # Additional denoising with residual
            ResidualDenoising(input_channels),
            nn.AvgPool1d(kernel_size=2, stride=2),

            # Final refinement
            nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_channels),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(50)  # Force output to 50 tokens
        )

    def forward(self, x):
        # Input shape: (batch_size, time_steps, features)
        # Need to transpose for CNN
        x = x.transpose(1, 2)  # Now: (batch_size, features, time_steps)

        # Process both streams
        left = self.left_extractor(x)  # Shape: (batch_size, features, 100)
        right = self.right_extractor(x)  # Shape: (batch_size, features, 50)

        # Transpose back to (batch_size, seq_len, features)
        left = left.transpose(1, 2)  # Shape: (batch_size, 100, features)
        right = right.transpose(1, 2)  # Shape: (batch_size, 50, features)

        return left, right


class ResidualDenoising(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out



class THAT_ENCODER(torch.nn.Module):
    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape):
        #
        ##
        super(THAT_ENCODER, self).__init__()
        #
        var_dim_feature = var_x_shape[-1]
        var_dim_time = var_x_shape[-2]
        var_dim_output = var_y_shape[-1]
        #    self.num_heads = 5

        # Replace the single output layer with multiple prediction heads
        #    self.layer_output = torch.nn.ModuleList([
        #        torch.nn.Linear(256 + 32, var_dim_output) for _ in range(self.num_heads)
        #   ])

        #
        ## ---------------------------------------- left ------------------------------------------
        #
        # Replace pooling with CNN feature extractor
        # self.feature_extractor = CNNFeatureExtractor(input_channels=var_dim_feature)

        # Gaussian encoding layers
        self.layer_left_gaussian = Gaussian_Position(var_dim_feature, 100)  # 100 tokens for left stream
        # self.layer_right_gaussian = Gaussian_Position(var_dim_feature, 50)  # 50 tokens for right stream

        # Left stream encoder

        var_num_left = 4
        var_dim_left = var_dim_feature
        self.layer_left_encoder = torch.nn.ModuleList([Encoder(var_dim_feature=var_dim_left,
                                                               var_num_head=10,
                                                               var_size_cnn=[1, 3, 5])
                                                       for _ in range(var_num_left)])
        #
        self.layer_left_norm = torch.nn.LayerNorm(var_dim_left, eps=1e-6)
        #
        self.layer_left_cnn_0 = torch.nn.Conv1d(in_channels=var_dim_left,
                                                out_channels=128,
                                                kernel_size=8)

        self.layer_left_cnn_1 = torch.nn.Conv1d(in_channels=var_dim_left,
                                                out_channels=128,
                                                kernel_size=16)
        #
        self.layer_left_dropout = torch.nn.Dropout(0.5)
        #
        ## --------------------------------------- right ------------------------------------------
        #
        # torch.nn.AvgPool1d(kernel_size=20, stride=20))
        # self.layer_left_pooling = torch.nn.AdaptiveAvgPool1d(270)

        #
        var_num_right = 4
        var_dim_right = 270
        self.layer_right_encoder = torch.nn.ModuleList([Encoder(var_dim_feature=var_dim_right,
                                                                var_num_head=10,
                                                                var_size_cnn=[1, 2, 3])
                                                        for _ in range(var_num_right)])
        #
        self.layer_right_norm = torch.nn.LayerNorm(var_dim_right, eps=1e-6)
        #
        self.layer_right_cnn_0 = torch.nn.Conv1d(in_channels=var_dim_right,
                                                 out_channels=16,
                                                 kernel_size=2)

        self.layer_right_cnn_1 = torch.nn.Conv1d(in_channels=var_dim_right,
                                                 out_channels=16,
                                                 kernel_size=4)
        #
        self.layer_right_dropout = torch.nn.Dropout(0.5)
        #
        ##
        self.layer_leakyrelu = torch.nn.LeakyReLU()
        #
        ##
        # self.layer_output = torch.nn.Linear(256 + 32, var_dim_output)

    #
    ##
    def forward(self, var_left, var_right):
        # Process through CNN feature extractor
        # var_left, var_right = self.feature_extractor(var_input)

        # Apply Gaussian position encoding
        var_left = self.layer_left_gaussian(var_left)  # Output: (batch_size, 100, features)
        # var_right = self.layer_right_gaussian(var_right)  # Output: (batch_size, 50, features)

        # Process left stream through transformer encoders
        for layer in self.layer_left_encoder:
            var_left = layer(var_left)
        var_left = self.layer_left_norm(var_left)

        # # Process right stream through transformer encoders
        # for layer in self.layer_right_encoder:
        #     var_right = layer(var_right)
        # var_right = self.layer_right_norm(var_right)

        # Concatenate the streams
        # var_t = torch.concat([var_left, var_right], dim=1)  # Output: (batch_size, 150, features)

        return var_left

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=270, nhead=5, num_decoder_layers=9, num_queries=5, dim_feedforward=512, dropout=0.1,
                 temp_cross_attention=1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Create object queries - learnable parameters
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))  # 10 object queries

        # Create decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            temp_cross_attention=temp_cross_attention,
            dropout=dropout
        )
        self.decoder_layers = nn.ModuleList([
            decoder_layer for _ in range(num_decoder_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output projection for classification and box prediction
        # Assuming num_classes is the number of activity classes
        self.class_embed = nn.Linear(d_model, 10)  # 10 activity classes

        # Create auxiliary outputs for each decoder layer + final output
        self.class_embed = nn.ModuleList([
            nn.Linear(d_model, 10) for _ in range(num_decoder_layers + 1)
        ])

    def forward(self, memory):
        """
        Args:
            memory: Output from encoder (B, 420, 270)
        Returns:
            outputs: List of output predictions from each decoder layer
        """
        B = memory.shape[0]

        # Initialize decoder input with zero queries
        tgt = torch.zeros_like(self.query_embed.unsqueeze(0).expand(B, -1, -1))

        # Get positional queries
        query_pos = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        # Store intermediate outputs
        intermediate = []

        # Run through decoder layers
        output = tgt
        for i, layer in enumerate(self.decoder_layers):
            output = layer(
                tgt=output,
                memory=memory,
                query_pos=query_pos
            )

            # Apply norm and get predictions for this layer
            normed_output = self.norm(output)
            pred = self.class_embed[i](normed_output)
            intermediate.append(pred)

        # Final layer norm and prediction
        output = self.norm(output)
        pred = self.class_embed[-1](output)
        intermediate.append(pred)

        return torch.stack(intermediate)  # Shape: [num_layers + 1, B, num_queries, num_classes]


class TemperatureMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, temperature=2.0, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.temperature = temperature

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True):
        # Regular attention computation
        attn_output, attn_weights = super().forward(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights
        )

        # Apply temperature scaling to attention output
        attn_output = attn_output / self.temperature

        return attn_output, attn_weights


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=270, nhead=8, dim_feedforward=2048, dropout=0.1, temp_cross_attention=1):
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = TemperatureMultiheadAttention(d_model, nhead, dropout=dropout,
                                                        temperature=temp_cross_attention, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, query_pos=None):
        # Self attention
        q = k = self.with_pos_embed(tgt, None)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=memory,
            value=memory
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed forward
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class DETR_MultiUser(nn.Module):
    def __init__(self, var_x_shape, var_y_shape, num_decoder_layers=12, temp_cross=1, num_queries=5):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(input_channels=var_x_shape[-1])
        # Encoder (your existing THAT_ENCODER)
        self.encoder = THAT_ENCODER(var_x_shape, var_y_shape)
        # Decoder
        self.decoder = TransformerDecoder(
            d_model=270,  # Matches encoder output feature dimension
            nhead=6,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1,
            num_queries=num_queries,
            temp_cross_attention=temp_cross
        )

    def forward(self, x):
        # Extracting Features
        var_left, var_right = self.feature_extractor(x)
        # Getting the memory to query from!
        memory = self.encoder(var_left, var_right)  # Shape: (B, 420, 270)

        # Pass through decoder to get predictions from all layers
        outputs_class = self.decoder(memory)  # Shape: [num_layers + 1, B, num_queries, num_classes]

        return outputs_class



class HungarianMatchingLoss(nn.Module):
    def __init__(self, cost_class_weight, aux_loss_weight, label_smoothing, class_imbalance_weight):
        """
        Hungarian Matching Loss for multi-user activity recognition with auxiliary losses
        Args:
            cost_class: Relative weight of classification loss
            aux_loss_weight: Weight for auxiliary losses from intermediate decoder layers
        """
        super().__init__()
        self.cost_class = cost_class_weight
        self.aux_loss_weight = aux_loss_weight

        weights = torch.ones(10)
        weights[-1] = class_imbalance_weight  # Last class gets lower weight since it appears more often
        weights = weights * (len(weights) / weights.sum())

        self.ce_loss = nn.CrossEntropyLoss(
            weight=weights.to(torch.device('cuda')),
            label_smoothing=label_smoothing
        )

    @torch.no_grad()
    def Hungarian_matching(self, outputs, targets):
        """
        Performs the matching between predictions and ground truth
        Args:
            outputs: Tensor of shape (batch_size, num_queries, num_classes)
            targets: Tensor of shape (batch_size, num_queries, num_classes)
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        bs, num_queries = outputs.shape[:2]

        # Compute classification cost matrix
        out_prob = outputs.softmax(-1)  # [batch_size, num_queries, num_classes]
        # Convert target one-hot to class indices
        tgt_ids = targets.argmax(-1)  # [batch_size, num_queries]

        # Compute cost matrix for all pairs of predictions and targets
        cost_class = -out_prob[:, :, tgt_ids.view(-1)].view(bs, num_queries, -1)

        # Final cost matrix
        C = self.cost_class * cost_class

        sizes = [num_queries] * bs
        indices = [linear_sum_assignment(c[i].cpu()) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def _get_layer_loss(self, pred, target, indices):
        """Helper to compute loss for a single layer's predictions"""
        losses = []
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            pred_i = pred[batch_idx][pred_idx]
            tgt_i = target[batch_idx][tgt_idx]
            loss = self.ce_loss(pred_i, tgt_i.argmax(-1))
            losses.append(loss.mean())
        return torch.stack(losses).mean()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: If auxiliary losses enabled: Tensor of shape [num_layers + 1, B, num_queries, num_classes]
                    Otherwise: Tensor of shape [B, num_queries, num_classes]
            targets: Tensor of shape [B, num_queries, num_classes]
        """
        # Check if we have auxiliary outputs
        if outputs.dim() == 4:  # Has auxiliary outputs [num_layers + 1, B, num_queries, num_classes]
            # Split predictions from different decoder layers
            aux_outputs = outputs[:-1]  # Predictions from intermediate layers
            outputs = outputs[-1]  # Predictions from final layer

            # Calculate loss for final predictions
            indices = self.Hungarian_matching(outputs, targets)
            final_loss = self._get_layer_loss(outputs, targets, indices)

            # Calculate auxiliary losses
            aux_losses = []
            for aux_output in aux_outputs:
                # For each layer, compute matching and loss
                aux_indices = self.Hungarian_matching(aux_output, targets)
                layer_loss = self._get_layer_loss(aux_output, targets, aux_indices)
                aux_losses.append(layer_loss)

            # Combine losses
            aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0)
            total_loss = final_loss + self.aux_loss_weight * aux_loss

            return total_loss

        else:  # No auxiliary outputs, just compute regular loss
            indices = self.Hungarian_matching(outputs, targets)
            return self._get_layer_loss(outputs, targets, indices)




def run_that_detr(data_train_x,
                     data_train_y,
                     data_test_x,
                     data_test_y,
                     var_repeat=10):
    """
    [description]
    : run WiFi-based model THAT_ENCODER_DECODER
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, labels to train model
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : var_repeat: int, number of repeated experiments
    [return]
    : result: dict, results of experiments
    """
    #
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    ##
    data_valid_x, data_test_x, data_valid_y, data_test_y = train_test_split(data_test_x, data_test_y,
                                                                            test_size=0.5,
                                                                            shuffle=True,
                                                                            random_state=39)

    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    ## shape for model
    var_x_shape, var_y_shape = data_train_x[0].shape, [data_train_y[0].shape[1]]
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    # data_test_set = TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y))
    data_valid_set = TensorDataset(torch.from_numpy(data_valid_x), torch.from_numpy(data_valid_y))

    #
    ##
    ## ========================================= Train & Evaluate =========================================
    result_ppp = []
    result_time_train = []
    result_time_test = []
    result_total_error = []
    result_precision = []
    result_recall = []
    result_f1_score = []

    #
    var_macs, var_params = get_model_complexity_info(DETR_MultiUser(var_x_shape, var_y_shape),
                                                     var_x_shape, as_strings=False)

    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    #

    for var_r in range(var_repeat):
        #
        ##
        var_mode = "multi_head"
        name_run = "Empty"
        if preset["pretrained_path"]:
            name_run = f"DETR_{var_r}_" + "_".join(preset["data"]["environment"]) + "_" + preset["transfer_scenario"]
        else:
            pretrained_state = "NPT"
            name_run = f"DETR_{var_r}_" + "_".join(preset["data"]["environment"]) + "_" + pretrained_state 
        print("Repeat", var_r)
        run = wandb.init(
            project="final_results",
            name= name_run,
            config=preset,
            reinit=True  # Allow multiple wandb.init() calls in the same process
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_detr = DETR_MultiUser(var_x_shape, var_y_shape, num_decoder_layers=preset["nn"]["num_decoder_layers"] ,temp_cross=preset["nn"]["cross_attention_temp"],
                                    num_queries=preset["nn"]["num_obj_queries"]).to(device)

        if preset.get("pretrained_path"):
            model_detr, param_groups = load_model_components(
                model_detr,
                preset["pretrained_path"],
                preset["nn"]["lr"],
                preset.get("transfer_scenario"),
                device
            )
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam(model_detr.parameters(),
                                         lr=preset["nn"]["lr"],
                                         weight_decay=preset["nn"]["weight_decay"])

        loss = HungarianMatchingLoss(
            cost_class_weight=preset["nn"]["loss"]["cost_class_weight"],
            aux_loss_weight=preset["nn"]["loss"]["aux_loss_weight"],
            label_smoothing=preset["nn"]["loss"]["label_smoothing"],
            class_imbalance_weight=preset["nn"]["loss"]["class_imbalance_weight"]
        )
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model=model_detr,
                                optimizer=optimizer,
                                loss=loss,
                                data_train_set=data_train_set,
                                data_test_set=data_valid_set,
                                var_threshold=preset["nn"]["threshold"],
                                var_batch_size=preset["nn"]["batch_size"],
                                var_epochs=preset["nn"]["epoch"],
                                device=device,
                                var_mode=var_mode)
        # Save model components based on scenario
        if preset.get("save_model"):
            save_model_components(preset, model_detr)
        #
        var_time_1 = time.time()
        #


        ## ---------------------------------------- Test ------------------------------------------
        #
        model_detr.load_state_dict(var_best_weight)
        #
        with torch.no_grad():
            predict_test_y = model_detr(torch.from_numpy(data_test_x).to(device))
        #
        # predict_test_y = torch.clamp(torch.round(predict_test_y), min=0, max=5).float()
        predict_test_y = predict_test_y.detach().cpu().numpy()
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##

        # data_test_y_c = data_test_y.sum(axis=1)
        dict_true_acc = performance_metrics(data_test_y, predict_test_y, var_mode=var_mode)
        wandb.log({
            "repeat": var_r,
            "train_time": var_time_1 - var_time_0,
            "test_time": var_time_2 - var_time_1,
            "TOTAL_TESTSET_ERROR": dict_true_acc['total_error'],
            "TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc['perfect_prediction_percentage'],
            "TOTAL_ACCURACY": dict_true_acc['accuracy'],
            "mean_count_error": dict_true_acc['mean_count_error'],
            "error_per_person_1": dict_true_acc['error_per_person'][0],
            "error_per_person_2": dict_true_acc['error_per_person'][1],
            "error_per_person_3": dict_true_acc['error_per_person'][2],
            "error_per_person_4": dict_true_acc['error_per_person'][3],
            "error_per_person_5": dict_true_acc['error_per_person'][4],
            "precision": dict_true_acc['precision'],
            "recall": dict_true_acc['recall'],
            "f1_score": dict_true_acc['f1_score']
        })
        print(" %.6fs" % (time.time() - var_time_1),
              "- Total Error %.6f" % dict_true_acc['total_error'],
              "-  perfect_prediction_percentage %.6f" % dict_true_acc['perfect_prediction_percentage'],
              )
        #
        #

        #
        result_ppp.append(dict_true_acc['perfect_prediction_percentage'])
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        result_total_error.append(dict_true_acc['total_error'])
        result_precision.append(dict_true_acc['precision'])
        result_recall.append(dict_true_acc['recall'])
        result_f1_score.append(dict_true_acc['f1_score'])

    wandb.log({
        "avg_accuracy": sum(result_ppp) / len(result_ppp),
        "avg_train_time": sum(result_time_train) / len(result_time_train),
        "avg_test_time": sum(result_time_test) / len(result_time_test),
        "avg_total_error": sum(result_total_error) / len(result_total_error),
        "avg_precision": sum(result_precision) / len(result_precision),
        "avg_recall": sum(result_recall) / len(result_recall),
        "avg_f1_score": sum(result_f1_score) / len(result_f1_score),
    })
    viz_stats = visualize_model_performance(
        y_pred=predict_test_y,
        y_true=data_test_y,
        var_mode=var_mode,
        save_dir=f'./visualizations/experiment_{var_r}_{var_mode}'
    )
    print("\nDetailed Performance Analysis:")
    print(f"Mean Error: {viz_stats['mean_error']:.4f} ± {viz_stats['error_std']:.4f}")
    print("\nClass-wise Mean Absolute Error:")
    for i, error in enumerate(viz_stats['class_wise_mae']):
        print(f"Class {i}: {error:.4f}")
    print(f"\nPerfect Predictions: {viz_stats['perfect_predictions'] * 100:.2f}%")
    wandb.finish()
    return dict_true_acc

