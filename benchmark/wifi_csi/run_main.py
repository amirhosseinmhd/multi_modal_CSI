"""
[file]          run.py
[description]   run WiFi-based models
"""
#
##
import json
import argparse
from logging import raiseExceptions

import numpy as np
from sklearn.model_selection import train_test_split
#
from model import *
from preset import preset
from load_data import load_data_x, load_data_y, encode_data_y
from utils import *
#
##
def mater_splitter(preset, var_task, var_model, var_users):
    env_data_x_train = []
    env_data_x_test = []
    env_data_y_train = []
    env_data_y_test = []
    for env in preset["data"]["environment"]:
        data_pd_y = load_data_y(preset["path"]["data_y"],
                                var_environment=[env],
                                var_wifi_band=preset["data"]["wifi_band"],
                                var_num_users=var_users)
        #
        var_label_list = data_pd_y["label"].to_list()
        #
        ## load CSI amplitude
        X = load_data_x(preset["path"]["data_x"], var_label_list)


        y = encode_data_y(data_pd_y, var_task)

        if var_model == "THAT_MULTI_HEAD":
            y = reduce_dataset(y)  # CHECKKKKKKKK HEREEEEEE
        elif var_model == "THAT_ENCODER" or var_model == "DETR":
            y = reduce_dataset(y, preset["nn"]["num_obj_queries"])  # CHECKKKKKKKK HEREEEEEE
        elif var_model == "THAT_COUNT_CONSTRAINED":
            y_red = reduce_dataset(y)
            y = y_red.sum(axis=1)
        else:
            pass

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            random_state=103)
        # np.random.randint()
        env_data_x_train.append(X_train)
        env_data_x_test.append(X_test)
        env_data_y_train.append(y_train)
        env_data_y_test.append(y_test)


    data_x_train = np.concatenate(env_data_x_train, axis = 0)
    data_x_test = np.concatenate(env_data_x_test, axis = 0)
    data_y_train = np.concatenate(env_data_y_train, axis = 0)
    data_y_test = np.concatenate(env_data_y_test, axis = 0)


    return data_x_train, data_x_test, data_y_train, data_y_test



def parse_args():
    """
    [description]
    : parse arguments from input
    """
    #
    ##
    var_args = argparse.ArgumentParser()
    #
    var_args.add_argument("--model", default = preset["model"], type = str)
    var_args.add_argument("--task", default = preset["task"], type = str)
    var_args.add_argument("--repeat", default = preset["repeat"], type = int)
    var_args.add_argument("--users", default="0, 1,2,3,4,5", type=str, help="Comma-separated list of user IDs")
    #
    return var_args.parse_args()

#
##
def run():
    """
    [description]
    : run WiFi-based models
    """
    #
    ## parse arguments from input
    var_args = parse_args()
    #
    var_task = var_args.task
    var_model = var_args.model
    var_repeat = var_args.repeat
    var_users = [u.strip() for u in var_args.users.split(',')]

    preset["repeat"] = 1 if not preset["pretrained_path"] else preset["repeat"] # if we want to pretrain the model we
    #                                                                           # need only one repeat

    # Ensuring there is no data leakage while doing splits.
    data_train_x, data_test_x, data_train_y, data_test_y = mater_splitter(preset, var_task, var_model, var_users)
    #

    ## a training set (80%) and a test set (30%)
    # data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data_x, data_y,
    #                                                                         test_size = 0.2,
    #                                                                         shuffle = True,
    #                                                                         random_state = 103)
    # #
    ## select a WiFi-based model
    if var_model == "ST-RF": run_model = run_strf
    #
    elif var_model == "MLP": run_model = run_mlp
    #
    elif var_model == "LSTM": run_model = run_lstm
    #
    elif var_model == "CNN-1D": run_model = run_cnn_1d
    #
    elif var_model == "CNN-2D": run_model = run_cnn_2d
    #
    elif var_model == "CLSTM": run_model = run_cnn_lstm
    #
    elif var_model == "ABLSTM": run_model = run_ablstm
    #
    elif var_model == "THAT": run_model = run_that
    #
    elif var_model == "SSL": run_model = run_ssl
    #
    elif var_model == "THAT_COUNT": run_model = run_that_count_pred
    #
    elif var_model == "THAT_MULTI_HEAD": run_model = run_that_multihead
    #
    elif var_model == "THAT_COUNT_CONSTRAINED": run_model = run_that_count_pred_contrained

    elif var_model == "THAT_ENCODER": run_model = run_that_decoder

    elif var_model == "DETR": run_model = run_that_detr

    else:
        raise Exception("Not valid name for model")




    #
    ## run WiFi-based model
    result = run_model(data_train_x, data_train_y,
                       data_test_x, data_test_y, var_repeat)
    #
    ##
    result["model"] = var_model
    result["task"] = var_task
    result["data"] = preset["data"]
    result["nn"] = preset["nn"]
    #
    print(result)
    #
    ## save results
    var_file = open(preset["path"]["save"], 'w')
    json.dump(result, var_file, indent=4, cls=NumpyEncoder)

#
##

if __name__ == "__main__":
    #
    ##
    run()