import logging
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

import data_preprocessing

LOG_FILENAME = "logfile.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

def linear_model_(housing_prepared, housing_labels):
    """ Linear model is fitted
    Arguments :
                housing_prepared and housing_labels
    Returns :   model
    """
    logging.info("Linear model.....")
    lin_reg = LinearRegression()
    model1 = lin_reg.fit(housing_prepared, housing_labels)
    return model1


def dtreg(housing_prepared, housing_labels):
    """ Decision tree model is fitted
    Arguments :
                housing_prepared and housing_labels
    Returns :   model
    """
    logging.info("Decision tree.....")
    tree_reg = DecisionTreeRegressor(random_state=42)
    model2 = tree_reg.fit(housing_prepared, housing_labels)
    return model2


def rnd_forest(housing_prepared, housing_labels):
    """ Random forest model is fitted
    Arguments :
                housing_prepared and housing_labels
    Returns :   model
    """
    logging.info("Random forest.....")

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)
    final_model = grid_search.best_estimator_

    return final_model
def save_pickle_file():
    logging.info("Pickle files are saved.....")
    linear, dt, rnd =model_train()
    Pkl_Filename1 = "Linear_Model.pkl"  
    with open(Pkl_Filename1, 'wb') as file:  
        pickle.dump(linear, file)
    Pkl_Filename2 = "DT_Model.pkl"  
    with open(Pkl_Filename2, 'wb') as file:  
        pickle.dump(dt, file)
    Pkl_Filename3 = "RF_gridsearch_Model.pkl"  
    with open(Pkl_Filename3, 'wb') as file:  
        pickle.dump(rnd, file)


def model_train():
    """ All models are trained
    Arguments :
                nothing
    Returns :   models
    """
    logging.info("model training.....")
    housing_prepared, housing_labels = data_preprocessing.data_preprocess()
    linear = linear_model_(housing_prepared, housing_labels)
    dt = dtreg(housing_prepared, housing_labels)
    rnd = rnd_forest(housing_prepared, housing_labels)
    return linear, dt, rnd



