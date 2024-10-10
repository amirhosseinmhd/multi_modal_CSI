import numpy as np


def calculate_matrix_absolute_error(y_true, y_pred):
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Calculate the absolute difference
    absolute_diff = np.abs(y_true - y_pred)

    # Calculate total error and error per sample
    total_error = np.sum(absolute_diff)
    perfect_predictions = np.sum(np.all(absolute_diff == 0, axis=1))
    total_samples = y_true.shape[0]
    perfect_prediction_percentage = (perfect_predictions / total_samples) * 100
    return {
        'total_error': total_error,
        'perfect_prediction_percentage': perfect_prediction_percentage,
    }








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
