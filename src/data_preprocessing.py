import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

LOG_FILENAME = "logfile.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """ downloads the housing data
    Arguments :
                housing_url,housing_path
    """
    logging.info("Fetch housing data.....")
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """ loads the housing data
    Arguments :
                housing_path
    Returns :   csv file
    """
    logging.info("Loads housing data.....")
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat(housing):
    """ creates a new column of income category
    Arguments :
                dataframe
    Returns :   modified dataframe
    """
    logging.info("Creating Income Category.....")
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


def housing_labels_(strat_train_set):
    """ creates copy of dataframe
    Arguments :
                strat_train_set
    Returns :   dataframe
    """
    logging.info("copy of dataset")
    housing_labels = strat_train_set["median_house_value"].copy()
    return housing_labels


def correlation_(housing):
    """ creates correlation matrix
    Arguments :
                housing
    Returns :   nothing
    """
    logging.info("Creating correlation matrix.....")
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


def stratified_split(housing):
    """ performs stratified split
    Arguments :
                housing
    Returns :   stratified train and stratified test
    """
    logging.info("Stratified split.....")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def imputer(housing):
    """ imputes missing values
    Arguments :
                housing
    Returns :   modified dataframe
    """
    logging.info("Imputation.....")
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    return housing_tr


def feature_eng2(housing_tr, housing):
    """ New features are added
    Arguments :
                housing_tr,housing
    Returns :   new dataframe housing_prepared
    """
    logging.info("Adding features.....")
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )
    return housing_prepared


def random_forest_test_Data(strat_test_set):
    """ prepares test data for random forest
    Arguments :
                strat_test_set
    Returns :   X_test,y_test
    """
    logging.info("Random forest.....")
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_num = X_test.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_test_num)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared = feature_eng2(X_test_prepared, X_test)
    return X_test_prepared, y_test


def rfdata():
    """ rf data function
    Arguments :
                Nothing
    Returns :   random forest test data
    """
    logging.info("rftest data")
    fetch_housing_data()
    housing = load_housing_data()
    housing = income_cat(housing)
    strat_train_set, strat_test_set = stratified_split(housing)
    return random_forest_test_Data(strat_test_set)


def data_preprocess():
    """ performs data preprocessing
    Arguments :
                nothing
    Returns :   housing_prepared,housing_labels
    """
    logging.info("data preprocessing.....")
    fetch_housing_data()
    housing = load_housing_data()
    housing = income_cat(housing)
    strat_train_set, strat_test_set = stratified_split(housing)
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_tr = imputer(housing)
    housing_prepared = feature_eng2(housing_tr, housing)
    return housing_prepared, housing_labels

data_preprocess()

