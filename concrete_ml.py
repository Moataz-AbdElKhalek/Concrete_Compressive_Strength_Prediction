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

# ANN Regression Model:
def NN_Regression(X,y, predict_sample=X[0]):
  # Divide the input dataset:
  X_train, X_test, y_train, y_test = dataset_divide(X,y)

  # Regression Model Training:
  model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(100,100,100), activation='relu',random_state=1, max_iter=1000)
  NN_Regressor = model.fit(X_train, y_train)
  score = NN_Regressor.score(X_test, y_test)
  prediction = NN_Regressor.predict(predict_sample)

  return score,prediction

def NN_Classification(X,y, predict_sample=X[0]):
  # Divide the input dataset:
  X_train, X_test, y_train, y_test = dataset_divide(X,y)

  # Classification Model Training:
  model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,100,100), activation='relu',random_state=1, max_iter=1000)
  NN_Classifier = model.fit(X_train, y_train)
  score = NN_Classifier.score(X_test, y_test)
  prediction = NN_Classifier.predict(predict_sample)

  return score,prediction

test_sample = X
current_score, prediction = NN_Regression(X, y1,test_sample)
print(current_score)
print(prediction)
matplotlib.pyplot.scatter(y1,prediction)
matplotlib.pyplot.show()
