#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing relevant modules
import numpy
import pandas as pd
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook as tqdm
import sklearn
import sklearn.model_selection
import sklearn.neural_network
import sklearn.ensemble
import sklearn.svm
import sklearn.preprocessing
import sklearn.metrics
import scipy.stats


# Ignoring warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load Dataset:
data_url = 'https://raw.githubusercontent.com/Moataz-AbdElKhalek/Concrete_Compressive_Strength_Prediction/main/dataset/Concrete_Dataset_Classification.csv'
dataset = pd.read_csv(data_url)

print(dataset.head(4))

# Descriptive statistics
print("\nDataset has {} rows and {} columns".format(dataset.shape[0],dataset.shape[1]))

print()
y = dataset['y']
print(y.head(4))
print(y.shape)
print()

X = dataset.drop(['y'], axis=1)
print(X.head(4))
print(X.shape)


# In[3]:


# Applying statistical Analysis on the data:
dataset.describe()


# # **Data Preprocessing**

# In[4]:


# Using Scikit-learn MaxMinScaler: 
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

# extract attributes and scale data to have Min = -1 and Max = 1 :
cols = X.columns
print('Data Attributes:\n', cols)
print('\nBefore Data Scaling:\n', X.head(4))
sc_X = scaler.fit_transform(X) # Fit scaler to data, then transform data to specified feature_range(-1,1)

# Turn the scaling results back into a dataframe :
sc_X_df = pd.DataFrame(sc_X, columns = cols)
X = sc_X_df
print('\nAfter Data Scaling:\n', X.head(4))

# Applying statistical Analysis on the data:
X.describe()


# # **SVM Model Optimization**
# # Using 10-fold Cross-Validation

# In[5]:


def SVM_CV(kernels,C_range,gamma_range):
    
  # Preparing the Model:
  model = sklearn.svm.SVC(random_state=1)

  # Determining Model Hyperparameters to be tested and optimized:
  paras = {'kernel':kernels, 'C':C_range, 'gamma':gamma_range}

  # Preparing Cross-Validation to be used to fit the Model and the Hyperparameters:
  # Using 10-fold Cross-Validation:
  gridCV = sklearn.model_selection.GridSearchCV(model, paras, cv=10, scoring='accuracy', verbose=10, n_jobs=-1)
  gridCV.fit(X, y)

  best_C = gridCV.best_params_['C']
  best_gamma = gridCV.best_params_['gamma']
  best_kernel = gridCV.best_params_['kernel']
  best_score = gridCV.best_score_
  results = gridCV.cv_results_

  return best_C, best_gamma, best_kernel, best_score, results


# In[6]:


best_C, best_gamma, best_kernel, best_score, results = SVM_CV(['rbf'],numpy.logspace(-5, 5, 11),numpy.logspace(-5, 5, 11))


# In[7]:


print('best_C =',best_C)
print('best_gamma =',best_gamma)
print('best_kernel =', best_kernel)
print('Cross-Validation Mean Best Score for the Model =',best_score)
print('\nCross-Validation Mean Test Scores\n', results['mean_test_score'])

for i in range(10):
  print('\nSplit_'+str(i+1)+' Scores\n',results['split'+str(i)+'_test_score'])
  print('best_score (Split_'+str(i+1)+') =', max(results['split'+str(i)+'_test_score']))


# # **Data Fixed Single Splitting**
# (70% Training and 30% Testing)

# In[8]:


# Dividing samples dataset into training and test datasets:
def dataset_divide(X, y):
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.70, random_state=1)
  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = dataset_divide(X,y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # **SVM Final Optimized Model**

# In[9]:


# SVM Classification Model:
def SVM_Classification(X,y, best_kernel='rbf', C_optimum = 100, gamma_optimum = 1):
  model = sklearn.svm.SVC(kernel= best_kernel, C = C_optimum, gamma = gamma_optimum, random_state=1)
  model.fit(X, y)

  return model


# In[10]:


# SVM Classification Model Training:
SVM_Model = SVM_Classification(X_train, y_train)

# SVM Classification Model Evaluation on Testing Data:
current_score = SVM_Model.score(X_test,y_test)
y_test_hat = SVM_Model.predict(X_test) # testing output

print('Score of Model Evaluation with Testing Data =', current_score)
# RMSE
rmse_test = sklearn.metrics.mean_squared_error(y_test, y_test_hat, squared=False)
print('rmse_test = ',rmse_test)

# Pearson's correlation
pcc_test = scipy.stats.pearsonr(y_test, y_test_hat)[0]
print ('pcc_test = ', pcc_test)

#Spearman's correlation
scc_test = scipy.stats.spearmanr(y_test, y_test_hat)[0]
print ('scc_test = ', scc_test)

matplotlib.pyplot.scatter(y_test,y_test_hat)
matplotlib.pyplot.show()

# SVM Classification Model Evaluation on Training Data:
current_score = SVM_Model.score(X_train,y_train)
y_train_hat = SVM_Model.predict(X_train) # testing output

print('Score of Model Evaluation with Training Data =', current_score)
# RMSE
rmse_train = sklearn.metrics.mean_squared_error(y_train, y_train_hat, squared=False)
print('rmse_train = ',rmse_train)
# Pearson's correlation
pcc_train = scipy.stats.pearsonr(y_train, y_train_hat)[0]
print ('pcc_train = ', pcc_train)

#Spearman's correlation
scc_train = scipy.stats.spearmanr(y_train, y_train_hat)[0]
print ('scc_train = ', scc_train)

matplotlib.pyplot.scatter(y_train,y_train_hat)
matplotlib.pyplot.show()

