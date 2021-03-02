import logging

import data_preprocessing
import train_model

LOG_FILENAME = "logfile.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
def predict():
    """ prediction of models
    Arguments :
                nothing
    Returns :   linear,dt,rf prediction
    """
    logging.info("Prediction.....")
    housing_prepared, housing_labels = data_preprocessing.data_preprocess()
    linear, dt, rnd = train_model.model_train()
    Linear_Model_prediction = linear.predict(housing_prepared)
    DT_Model_prediction = dt.predict(housing_prepared)
    X_test_prepared, y_test = data_preprocessing.rfdata()
    RF_Model_prediction = rnd.predict(X_test_prepared)
    return Linear_Model_prediction, DT_Model_prediction, RF_Model_prediction
