import numpy as np
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch

import os


def process_predictions(y_pred, y_true, var_threshold=0.5):
    """
    Process activity predictions to count activities above threshold.

    Args:
        y_pred: numpy array of shape (num_samples, 6, 9) containing probabilities
        y_true: numpy array of shape (num_samples, 6, 9) containing true values
        var_threshold: minimum probability threshold for counting an activity

    Returns:
        y_pred_processed: summed activity counts (num_samples, 9)
        y_true_processed: summed true activity counts (num_samples, 9)
        batch_size: number of samples in batch
    """
    # Get the indices of max probabilities for each user
    max_indices = np.argmax(y_pred, axis=2)  # Shape: (num_samples, 6)

    # Get the corresponding maximum probabilities
    max_probs = np.take_along_axis(y_pred, np.expand_dims(max_indices, axis=2), axis=2)
    max_probs = max_probs.squeeze(axis=2)  # Shape: (num_samples, 6)

    # Create a mask for probabilities above threshold
    above_threshold = max_probs > var_threshold  # Shape: (num_samples, 6)

    # Create one-hot encoded matrix for the predicted activities
    y_pred_one_hot = np.zeros_like(y_pred)  # Shape: (num_samples, 6, 9)
    batch_indices = np.arange(y_pred.shape[0])[:, None]
    user_indices = np.arange(y_pred.shape[1])[None, :]
    y_pred_one_hot[batch_indices, user_indices, max_indices] = above_threshold

    # Sum up the activities across users
    y_pred_processed = y_pred_one_hot.sum(axis=1)  # Shape: (num_samples, 9)
    y_true_processed = y_true.sum(axis=1)  # Shape: (num_samples, 9)

    batch_size = y_true.shape[0]

    return y_pred_processed, y_true_processed, batch_size




def calculate_matrix_absolute_error(y_true, y_pred, var_mode = "multi_head", var_threshold = 0.5):
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if var_mode == "count_classification_withConstrain":
        batch_size, num_classes = y_pred.shape
    elif var_mode == "multi_head":
        batch_size, num_heads, num_classes = y_pred.shape
        y_pred_indices = np.argmax(y_pred, axis=-1)
        y_pred = np.eye(num_classes)[y_pred_indices] # this gives us one hot encoded version of it.
        y_pred = y_pred.sum(axis=1)  # summing along the columns, this should give us count of each activity
        y_true = y_true.sum(axis=1)
        y_pred = y_pred[:, :-1]
        y_true = y_true[:, :-1]

    elif var_mode == "count_classification":
        batch_size, num_classes = y_pred.shape
        y_pred = np.clip(np.round(y_pred), a_min=0, a_max=5)
    elif var_mode == "baseline":
        y_pred = (1 / (1 + np.exp(-y_pred))).astype(float)
        y_true = y_true.reshape(y_true.shape[0], -1, 9)
        y_pred = y_pred.reshape(y_true.shape[0], y_true.shape[1], y_true.shape[2])

        # Process predictions with threshold
        y_pred, y_true, batch_size = process_predictions(y_pred, y_true, var_threshold=0.5)
        batch_size = y_true.shape[0]
    else:
        raise ValueError(f"Unsupported var_mode: {var_mode}")

    # Calculate the absolute difference
    absolute_diff = np.abs(y_true - y_pred)
    acc_bysample = (1*absolute_diff == 0).sum(axis=1)/absolute_diff.shape[1]
    acc = acc_bysample.mean()
    std = acc_bysample.std()
    # Calculate total error and error per sample
    perfect_predictions = np.sum(np.all(absolute_diff == 0, axis=1))
    perfect_prediction_percentage = (perfect_predictions / batch_size) * 100
    total_error = np.sum(absolute_diff)/batch_size
    return {
        'total_error': total_error,
        'perfect_prediction_percentage': perfect_prediction_percentage,
        'accuracy': acc
    }

def reduce_dataset(data):
    new_data = []
    zero = np.zeros((5, 1))

    for sample in data:
        # Count non-zero rows-pp
        legend_non_zero = sample.sum(axis=1)
        new_sample = np.delete(sample, (legend_non_zero == 0).argmax(), axis=0)
        new_sample = np.hstack((new_sample, zero))
        legend_non_zero = new_sample.sum(axis=1)
        new_sample[legend_non_zero == 0, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        new_data.append(new_sample)
    return np.array(new_data)

def visualize_model_performance(y_pred, y_true, save_dir="./visualizations",var_mode="multi_head"):
    """
    Creates and saves various visualizations of model performance
    y_pred: numpy array [batch_size, 10] (predicted counts)
    y_true: numpy array [batch_size, 10] (true counts)
    """
    print(var_mode)
    if var_mode == "count_classification_withConstrain":
        pass
    elif var_mode == "multi_head":
        batch_size, num_heads, num_classes = y_pred.shape
        y_pred_indices = np.argmax(y_pred, axis=-1)
        y_pred = np.eye(num_classes)[y_pred_indices] # this gives us one hot encoded version of it.
        y_pred = y_pred.sum(axis=1)  # summing along the columns, this should give us count of each activity
        y_true = y_true.sum(axis=1)

    elif var_mode == "count_classification":
        batch_size, num_classes = y_pred.shape
        y_pred = np.clip(np.round(y_pred), a_min=0, a_max=5)
    elif var_mode == "baseline":
        y_pred = (1 / (1 + np.exp(-y_pred)) > 0.5).astype(float)
        y_true = y_true.reshape(y_true.shape[0], -1, 9)
        y_pred = y_pred.reshape(y_true.shape[0], y_true.shape[1], y_true.shape[2])
        y_pred = y_pred.sum(axis=1) # summing along the columns, this should give us count of each activity
        y_true = y_true.sum(axis=1)
    else:
        raise ValueError(f"Unsupported var_mode: {var_mode}")
    os.makedirs(f"{save_dir}", exist_ok=True)

    # 1. Distribution of Predictions vs Ground Truth
    plt.figure(figsize=(15, 5))

    # Plot for each class
    for i in range(int(y_pred.shape[1])):
        plt.subplot(2, 5, i + 1)
        plt.hist(y_true[:, i], alpha=0.5, label='Ground Truth', bins=range(7))
        plt.hist(y_pred[:, i], alpha=0.5, label='Predicted', bins=range(7))
        plt.title(f'Class {i}')
        plt.xlabel('Count')
        plt.ylabel('Frequency')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/count_distributions_{var_mode}.png')
    plt.close()

    # 2. Confusion Matrix for each class
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(int(y_pred.shape[1])):
        ax = axes[i // 5, i % 5]
        cm = confusion_matrix(y_true[:, i], np.round(y_pred[:, i]))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_title(f'Class {i}')
        ax.set_xlabel('Predicted Count')
        ax.set_ylabel('True Count')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrices_{var_mode}.png')
    plt.close()

    # 3. Error Distribution
    plt.figure(figsize=(10, 6))
    errors = np.abs(y_pred - y_true).mean(axis=1)
    plt.hist(errors, bins=30)
    plt.title('Distribution of Mean Absolute Error per Sample')
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Frequency')
    plt.savefig(f'{save_dir}/error_distribution_{var_mode}.png')
    plt.close()

    # 4. Class-wise Error Analysis
    plt.figure(figsize=(10, 6))
    class_errors = np.abs(y_pred - y_true).mean(axis=0)
    plt.bar(range(int(y_pred.shape[1])), class_errors)
    plt.title('Mean Absolute Error by Class')
    plt.xlabel('Class')
    plt.ylabel('Mean Absolute Error')
    plt.savefig(f'{save_dir}/class_errors_{var_mode}.png')
    plt.close()

    # 5. Prediction vs Ground Truth Scatter
    plt.figure(figsize=(10, 10))
    for i in range(int(y_pred.shape[1])):
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.1, label=f'Class {i}')
    plt.plot([0, 5], [0, 5], 'r--')  # Perfect prediction line
    plt.xlabel('True Count')
    plt.ylabel('Predicted Count')
    plt.title('Predicted vs True Counts')
    plt.legend()
    plt.savefig(f'{save_dir}/prediction_scatter_{var_mode}.png')
    plt.close()

    # Return summary statistics
    return {
        'class_wise_mae': class_errors.tolist(),
        'mean_error': errors.mean(),
        'error_std': errors.std(),
        'perfect_predictions': (np.abs(y_pred - y_true) < 0.5).all(axis=1).mean()
    }

