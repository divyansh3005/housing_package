# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

Overview on How to use the Package
==================================

1. The package can be installed by running the following command inside package, the root directory of the package.
            pip install -e ./
2. A env.yml file is also provided which can be used to create a new python virtual environment with the required packages installed. Run the following command inside the mypackage, the root directory.
            conda env create -f env.yml
3. Once the package is installed, the module package can be imported and the functions can be called from it.
4. There are 4 functions provided in the package. They are

  | data_preprocessing()
  |
  | train_model()
  |
  | prediction()
  |
  | score_eval()

5. data_preprocessing() creates two datasets housing_prepared and housing_labels which are used for training the model.

6. train_model() is used for training three types of model.
   | Linear Model
   |
   | Decision Tree Model
   |
   | Random Forest Model
   |
   All three models are being saved as pickle files.

7. prediction() function predicts on test data for all three models.

8. score_eval() function displays the Root Mean Square Error for all three models.

9. To use the functions import the package and call the functions from it as follows.

  | import package
  |
  | package.data_preprocessing()
  |
  | package.train_model()
  |
  | package.prediction()
  |
  | package.score_eval()

11. A test is also available , for testing the rmse for each models.

  | To test the functions, run the below command in the command line from any directory.

            pytest --pyargs package

Setup procedure
===============

1. If python is not installed in your linux system, follow the instructions in the below link to install it.
   https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
2. To test the package, install the package by following step 1 in "Overview on How to use the Package" section.
3. It will be healthy to first create a virtual environment as mentioned in step 2 of "Overview on How to use the Package" section and then install the package.
4. If you opted to create a new environment, make sure the virtual environment is successfully created with the required packages by running the following commands in package root directory. Make sure to use this environment only when using the package.
            conda activate environment
            python test_install.py
5. Once the package is successfully installed, you can import the package into any python project using the command
            import package
6. For using the functions in the package, you can call the functions from the package as follows

  | package.data_preprocessing()
  |
  | package.train_model()
  |
  | package.prediction()
  |
  | package.score_eval()

Welcome to Median_House_Value's documentation!
==============================================

package main
==============
.. automodule:: package.main
   :members:

package score_eval
===============
.. automodule:: package.score_eval
   :members:

package prediction
====================
.. automodule:: mypackage.prediction
   :members:

package model_score
=====================
.. automodule:: package.train_model
   :members:

package data_preprocessing
====================
.. automodule:: package.data_preprocessing
   :members:


