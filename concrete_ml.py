# Commented out IPython magic to ensure Python compatibility.
# Importing relevant modules
import sklearn.neural_network
import numpy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
# %matplotlib inline
import sklearn

# Ignoring warnings
import warnings
warnings.filterwarnings("ignore")

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

# Applying statistical Analysis on the data:
dataset.describe()

# Dividing samples dataset into training and test datasets:
def dataset_divide(X, y):
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.70, random_state=1)
  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = dataset_divide(X,y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# ANN Classification Model:
def NN_Classification(X,y, max_iter=200, alpha=0.01668, hidden_layer_sizes=[39, 39, 39, 39, 39, 39]):
  model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu',random_state=1, max_iter=max_iter,alpha=alpha)
  model.fit(X, y)

  return model

# ANN Classification Model Training:
NN_Model = NN_Classification(X_train, y_train)

# ANN Classification Model Testing:
current_score = NN_Model.score(X_test,y_test)
y_test_hat = NN_Model.predict(X_test) # testing output]

print(current_score)
matplotlib.pyplot.scatter(y_test,y_test_hat)
matplotlib.pyplot.show()

#def learning_curve_maxiters(X, y, filename=0):
#  return 0

max_iters = numpy.arange(50,1000,10) # 1-D numpy array of max_iterations from 50 to 2000 with step 50
scores_iter = numpy.array([]) # 1-D numpy array of scores

for current_max_iter in max_iters:
  NN_Model = NN_Classification(X_train, y_train, max_iter=current_max_iter)
  current_score = NN_Model.score(X_test,y_test)
  scores_iter = numpy.append(scores_iter,current_score)

#learning_curve_maxiters(X_train,y_train)

best_score_iter_index = numpy.argmax(scores_iter)
best_score_iter = scores_iter[best_score_iter_index]
best_max_iters = max_iters[best_score_iter_index]

print('best_score_iter =', best_score_iter)
print('best_score_iter_index =', best_score_iter_index)
print('best_max_iters =', best_max_iters)

matplotlib.pyplot.plot(max_iters,scores_iter)
#matplotlib.pyplot.show()
matplotlib.pyplot.savefig('max_iterations_scores.png')

#def learning_curve_alpha(X, y, filename=0):
#  return 0

alphas = numpy.logspace(-100, 2, 10000) # 1-D numpy array of max_iterations from 50 to 2000 with step 50
scores_alpha = numpy.array([]) # 1-D numpy array of scores

for current_alpha in alphas:
  NN_Model = NN_Classification(X_train, y_train, alpha=current_alpha)
  current_score = NN_Model.score(X_test,y_test)
  scores_alpha = numpy.append(scores_alpha,current_score)

#learning_curve_alpha(X_train,y_train)

best_score_alpha_index = numpy.argmax(scores_alpha)
best_score_alpha = scores_alpha[best_score_alpha_index]
best_alpha = alphas[best_score_alpha_index]

print('best_score_alpha =', best_score_alpha)
print('best_score_alpha_index =', best_score_alpha_index)
print('best_alpha =', best_alpha)

best_score_alpha_indeces = numpy.where(scores_alpha == best_score_alpha)
print('best_score_alpha_indeces =', best_score_alpha_indeces)
print('best_alphas =', alphas[best_score_alpha_indeces])

matplotlib.pyplot.plot(alphas,scores_alpha)
#plt.xlim(0, 1)
#matplotlib.pyplot.show()
matplotlib.pyplot.savefig('alpha_scores.png')

def learning_curve_hidden(X, y, filename=0):
  return 0

scores_hidden = numpy.array([]) # 1-D numpy array of scores
layers = []
number_of_layers = 100
number_of_nodes_per_layers = 100

for n in range(number_of_nodes_per_layers):
  temp = numpy.array([[n+1]])
  for m in range(number_of_layers):
    hidden = numpy.repeat(temp, repeats=m+1, axis=1)
    layers.append(hidden.tolist())

for i in range(number_of_layers*number_of_nodes_per_layers):
  NN_Model = NN_Classification(X_train, y_train, hidden_layer_sizes=layers[i][0])
  current_score = NN_Model.score(X_test,y_test)
  scores_hidden = numpy.append(scores_hidden,current_score)

#learning_curve_hidden(X_train,y_train)

best_score_hidden_index = numpy.argmax(scores_hidden)
best_score_hidden = scores_hidden[best_score_hidden_index]
best_hidden = layers[best_score_hidden_index]

print('best_score_hidden =', best_score_hidden)
print('best_score_hidden_index =',best_score_hidden_index)
print('best_hidden_layer =',best_hidden)

best_score_hidden_indeces = numpy.where(scores_hidden == best_score_hidden)
print('best_score_hidden_indeces =', best_score_hidden_indeces)
for i in best_score_hidden_indeces[0]:
  print('best_hidden_layers =', layers[i])

layer_index = range(number_of_layers*number_of_nodes_per_layers)
matplotlib.pyplot.plot(layer_index,scores_hidden)
#matplotlib.pyplot.show()
matplotlib.pyplot.savefig('hidden_layers_score.png')

"""# SVM"""

def study_SVM(C_range,kernel,gamma):
    
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    
    scores = np.zeros((len(gamma),len(C_range)))
    
    for i in range (0,len(gamma)):
        for j in range (0,len(C_range)):
            clf = svm.SVC(C=C_range[j],kernel =kernel,gamma=gamma[i],random_state=1)
            clf.fit(X_train,y_train)
            clf.predict(X_test)
            scores[i,j]=clf.score(X_test,y_test)
    
    #Finding best parameters
    best_C, best_gamma = 0, 0
    model = sklearn.svm.SVC(kernel=kernel,random_state=1)
    paras = {'C':C_range, 'gamma':gamma}
    clf_1 = GridSearchCV(model, paras)
    clf_1.fit(X_train, y_train)
    clf_1.predict(X_test)
    best_C=clf_1.best_params_['C']
    best_gamma=clf_1.best_params_['gamma']
            
    print('---------Max.score---------')
    print(np.amax(scores))
    print('---------Best_C---------')
    print(best_C)
    print('---------Best_gamma---------')
    print(best_gamma)
    print('---------Full.scores---------')
    return scores

study_SVM([10**p for p in range(-5, 5, 1)],'rbf',[10**p for p in range(-5, 5, 1)])
