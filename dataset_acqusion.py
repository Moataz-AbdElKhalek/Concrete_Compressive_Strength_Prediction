# import relevant modules
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy
#import seaborn as sns
import sklearn
#import imblearn

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load Dataset:
dataset = pd.read_csv("Project_Dataset_Classification.csv")

print(dataset.head(4))

# Descriptive statistics
print("Dataset has {} rows & {} columns".format(dataset.shape[0],dataset.shape[1]))
dataset.describe()

y = dataset['y']
print(y.head(4))
print(y.shape)

X = dataset.drop(['y'], axis=1)
print(X.head(4))
print(X.shape)

def dataset_divide(X, y):
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
  return X_train, X_test, y_train, y_test

print(dataset_divide(X, y))
