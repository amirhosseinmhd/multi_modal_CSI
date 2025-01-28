import os
import sys
from copy import deepcopy


def load_preset(preset_file):
    """Load the preset configuration from a Python file."""
    sys.path.append(os.path.dirname(preset_file))
    module_name = os.path.splitext(os.path.basename(preset_file))[0]
    __import__(module_name)
    return deepcopy(sys.modules[module_name].preset)


def update_config_from_env(config):
    """Update configuration based on environment variables."""
    if 'DATA_PATH'in os.environ:
        config['path']['data_x'] = str(os.environ['DATA_PATH']) + "/wifi_csi/amp"
        config['path']['data_y'] = str(os.environ['DATA_PATH']) + "/annotation.csv"

    if 'LEARNING_RATE' in os.environ:
        config['nn']['lr'] = float(os.environ['LEARNING_RATE'])
    if 'BATCH_SIZE' in os.environ:
        config['nn']['batch_size'] = int(os.environ['BATCH_SIZE'])
    if 'NUM_EPOCHS' in os.environ:
        config['nn']['epoch'] = int(os.environ['NUM_EPOCHS'])
    if 'NUM_DECODER_LAYERS' in os.environ:
        config['nn']['num_decoder_layers'] = int(os.environ['NUM_DECODER_LAYERS'])
    if 'DIM_FFN' in os.environ:
        config['nn']['dim_FFN'] = int(os.environ['DIM_FFN'])
    if 'NUM_QUERIES' in os.environ:
        config['nn']['num_obj_queries'] = int(os.environ['NUM_QUERIES'])
    if "AUX_LOSS" in os.environ:
        config['nn']['loss']['aux_loss_weight'] = float(os.environ['AUX_LOSS'])
    if "CLASS_IMBALANCE_WEIGHT" in os.environ:
        config['nn']['loss']['class_imbalance_weight'] = float(os.environ['CLASS_IMBALANCE_WEIGHT'])
    if "LABEL_SMOOTHING" in os.environ:
        config['nn']['loss']['label_smoothing'] = float(os.environ['LABEL_SMOOTHING'])

    if 'MODEL_TYPE' in os.environ:
        config['model'] = os.environ['MODEL_TYPE']

    if 'ENVIRONMENTS_EXP' in os.environ:
        # Split by comma and strip whitespace
        environments = [env.strip() for env in os.environ['ENVIRONMENTS_EXP'].split(',')]
        config['data']['environment'] = environments
    return config


def format_value(value):
    """Format Python values to their proper string representation."""
    if value is None:
        return "None"
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, (list, tuple)):
        return str(value)
    elif isinstance(value, dict):
        formatted_dict = {k: format_value(v) for k, v in value.items()}
        return "{\n" + ",\n".join(f"        '{k}': {v}" for k, v in formatted_dict.items()) + "\n    }"
    return str(value)


def save_config(config, output_file):
    """Save the modified configuration in Python format."""
    with open(output_file, 'w') as f:
        f.write("preset = {\n")
        for key, value in config.items():
            formatted_value = format_value(value)
            f.write(f"    '{key}': {formatted_value},\n")
        f.write("}\n")


if __name__ == "__main__":
    preset_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load and modify configuration
    config = load_preset(preset_file)
    modified_config = update_config_from_env(config)

    # Save modified configuration
    save_config(modified_config, output_file)