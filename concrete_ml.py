import sklearn.neural_network
import numpy
import matplotlib.pyplot
import itertools
import warnings
warnings.filterwarnings("ignore")

# Dividing samples dataset into training and test datasets:
### I think we better remove the function as by this way with each call we get different datasets for training and testing resulting in different outputs for each model.
def dataset_divide(X, y):
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
  return X_train, X_test, y_train, y_test


