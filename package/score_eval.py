import logging

import numpy as np
# from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import data_preprocessing
import prediction

LOG_FILENAME = "logfile.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


def pred_eval():
    """ Evaluation of models
    Arguments :
                nothing
    Returns :   evaluation metrics lin_rmse,dt_rmse,final_rmse
    """
    _, housing_labels = data_preprocessing.data_preprocess()

    (
        Linear_Model_prediction,
        DT_Model_prediction,
        RF_Model_prediction,
    ) = prediction.predict()
    lin_mse = mean_squared_error(housing_labels, Linear_Model_prediction)
    lin_rmse = np.sqrt(lin_mse)
    # lin_mae = mean_absolute_error(housing_labels, Linear_Model_prediction)
    tree_mse = mean_squared_error(housing_labels, DT_Model_prediction)
    tree_rmse = np.sqrt(tree_mse)
    _, y_test = data_preprocessing.rfdata()
    final_mse = mean_squared_error(y_test, RF_Model_prediction)
    final_rmse = np.sqrt(final_mse)
    logging.info("prediction evaluation.....")
    return lin_rmse, tree_rmse, final_rmse
