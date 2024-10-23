import numpy as np


def calculate_matrix_absolute_error(y_true, y_pred, var_mode = "multi_head", var_threshold = 0.5):
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if var_mode == "multi_head":
        batch_size, num_heads, num_classes = y_pred.shape
        y_true = y_true[:,:, :-1]
        y_pred = y_pred[:,:, :-1]
        y_pred_indices = np.argmax(y_pred, axis=-1)
        y_pred = np.eye(num_classes-1)[y_pred_indices] # this gives us one hot encoded version of it.
        y_pred = y_pred.sum(axis=1)  # summing along the columns, this should give us count of each activity
        y_true = y_true.sum(axis=1)

    if var_mode == "count_classification":
        batch_size, num_classes = y_pred.shape
        y_pred = np.clip(np.round(y_pred), a_min=0, a_max=5)

    if var_mode == "baseline":
        y_pred = (1 / (1 + np.exp(-y_pred)) > var_threshold).astype(float)
        y_true = y_true.reshape(y_true.shape[0], -1, 9)
        y_pred = y_pred.reshape(y_true.shape[0], y_true.shape[1], y_true.shape[2])
        y_pred = y_pred.sum(axis=1) # summing along the columns, this should give us count of each activity
        y_true = y_true.sum(axis=1)
        batch_size = y_true.shape[0]

    # Calculate the absolute difference
    absolute_diff = np.abs(y_true - y_pred)
    acc_bysample = (1*absolute_diff == 0).sum(axis=1)/absolute_diff.shape[1]
    acc = acc_bysample.mean()
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

    # error_per_sample = np.sum(absolute_diff, axis=1)
    # # print(error_per_sample.shape)
    # # Calculate average error per activity
    # avg_error_per_activity = np.mean(absolute_diff, axis=0)
    # # print(avg_error_per_activity.shape)
    # # Calculate the number of perfectly predicted samples

    #
    # # Calculate the total number of people correctly placed
    # correct_placements = np.sum(y_true == y_pred)
    # total_placements = np.sum(y_true)  # Total number of people across all samples and activities
    # correct_placement_percentage = (correct_placements / total_placements) * 100 if total_placements > 0 else 100

# 'error_per_sample': error_per_sample,
# 'avg_error_per_activity': avg_error_per_activity,
# 'correct_placement_percentage': correct_placement_percentage
