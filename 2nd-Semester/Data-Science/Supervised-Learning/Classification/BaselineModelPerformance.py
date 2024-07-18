# Supervised Learning: Classification Models

# Classification models are used to assign items to a discrete group or class based on a specific set of features. Each model has its own strengths and weaknesses for a given data set. Choosing a data classification model is also closely tied to the business case and a solid understanding of what you are trying to accomplish.

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

path = '/home/aspphem/Desktop/MCE/DataScience/T5/Scripts/Figures' 

if not os.path.exists(path):
    os.makedirs(path)

X_train = pd.read_csv('/home/aspphem/Desktop/MCE/DataScience/T5/SuppMaterial/mnist/mnist_Xtrain.csv', header = None).to_numpy() # training data set 
y_train = pd.read_csv('/home/aspphem/Desktop/MCE/DataScience/T5/SuppMaterial/mnist/mnist_Ytrain.csv', header = None).to_numpy() # set of labels to all the data in train set

X_test = pd.read_csv('/home/aspphem/Desktop/MCE/DataScience/T5/SuppMaterial/mnist/mnist_Xtest.csv', header = None).to_numpy() # test data set
y_test = pd.read_csv('/home/aspphem/Desktop/MCE/DataScience/T5/SuppMaterial/mnist/mnist_Ytest.csv', header = None).to_numpy() # set of labels to all the data in test set

print(X_train.shape) # X train dimensions (60000, 784)
print(y_train.shape) # y train dimensions (60000, 1)

print(X_test.shape) # X train dimensions (10000, 784)
print(y_test.shape) # y train dimensions (10000, 1)

X_train = X_train[0:10000,:]
y_train = y_train[0:10000,:]

X_test = X_test[0:2000,:]
y_test = y_test[0:2000,:]

y = y_train.ravel()
y_train = np.array(y).astype(int)

# Logistic Regression

# Logistic regression is a generalized linear model: Linear regression gives a continuous value of output y for a given input x. Whereas, logistic regression gives a continuous value of P(Y = 1) for a given input X, which is later converted to Y = 0 or Y = 1 based on a threshold value.

# Logistic regression uses the Sigmoid function to predict the output class label for a given input.

logistic_regression = LogisticRegression(fit_intercept = True, multi_class = 'auto', penalty = 'l2', solver = 'saga', max_iter = 1000, dual = False, random_state = 0) # SAGA solver is a variant of SAG (Stochastic Average Gradient) that supports both L1 and L2 penalization and is also suitable for very large datasets

logistic_regression.fit(X_train, y_train) # fit the model according to the given training data

predictions = logistic_regression.predict(X_test) # predict class for X
print(predictions[0:9])
print(np.transpose(y_test[0:9]))
print("Logistic Regression Classifier: Training Score {}, Test Score {}".format(logistic_regression.score(X_train, y_train), logistic_regression.score(X_test, y_test))) # mean accuracy on the given data and labels

_, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (12, 6))
axes = axes.flatten()

for ax, image, prediction in zip(axes, X_test, predictions):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap = plt.cm.gray_r, interpolation = 'nearest')
    ax.set_title(f'Prediction: {prediction}')

plt.savefig(os.path.join(path, 'PredictionsLogRegression.png'))
plt.close()

cm = metrics.confusion_matrix(y_test, predictions) 
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap='Blues')

plt.savefig(os.path.join(path, 'ConfusionMatrixLogRegression.png'))
plt.close()

print(f"Classification Report {logistic_regression} Classifier:\n"
      f"{metrics.classification_report(y_test, predictions)}\n") 

# Neural Networks

mlp = MLPClassifier(activation = 'tanh', solver = 'adam', alpha = 0.32, random_state = 0, max_iter = 300) # multi-layer perceptron classifier; activation function for the hidden layer hyperbolic function (tanh); solver 'adam' works pretty well oan large datasets in terms of both training time and validation score; alpha is a regularization term (increasing alpha (fix high variance) results in a decision boundary plot that appears with lesser curvatures; decreasing alpha (fix high bias) results in a more complicated decision boundary)

mlp.fit(X_train, y_train) # fit the model to data matrix X and targets y

predictions = mlp.predict(X_test) # predict class for X
print(predictions[0:9])
print(np.transpose(y_test[0:9]))
print("MLP Classifier: Training Score {}, Test Score {}".format(mlp.score(X_train, y_train), mlp.score(X_test, y_test))) # mean accuracy on the given data and labels

_, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (12, 6))
axes = axes.flatten()

for ax, image, prediction in zip(axes, X_test, predictions):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap = plt.cm.gray_r, interpolation = 'nearest')
    ax.set_title(f'Prediction: {prediction}')

plt.savefig(os.path.join(path, 'PredictionsMLP.png'))
plt.close()

cm = metrics.confusion_matrix(y_test, predictions)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap='Blues')

plt.savefig(os.path.join(path, 'ConfusionMatrixMLP.png'))
plt.close()

print(f"Classification Report {mlp} Classifier:\n"
      f"{metrics.classification_report(y_test, predictions)}\n")

# Support Vector Machines (SVM)

# Classification of data by finding an optimal hyperplane for linearly separable patterns. Extend to patterns that are not linearly separable by transformations of original data mapping it into a new space through a kernel function.

# Support vectors are the data points that lie closest to the decision surface (hyperplane). Thus optimal hyperplane maximize margin between support vectors (classes).

svm = svm.SVC(C = 1.0, kernel = 'rbf', decision_function_shape = 'ovo', random_state = 0) # multiclass support handled according to a one-against-one strategy

svm.fit(X_train, y_train) # fit the SVM model according to the given training data

predictions = svm.predict(X_test) # predict class for X
print(predictions[0:9])
print(np.transpose(y_test[0:9]))
print("SVM Classifier: Training Score {}, Test Score {}".format(svm.score(X_train, y_train), svm.score(X_test, y_test))) # mean accuracy on the given data and labels

_, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (12, 6))
axes = axes.flatten()

for ax, image, prediction in zip(axes, X_test, predictions):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap = plt.cm.gray_r, interpolation = 'nearest')
    ax.set_title(f'Prediction: {prediction}')

plt.savefig(os.path.join(path, 'PredictionsSVM.png'))
plt.close()

cm = metrics.confusion_matrix(y_test, predictions)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap='Blues')

plt.savefig(os.path.join(path, 'ConfusionMatrixSVM.png'))
plt.close()

print(f"Classification Report {svm} Classifier:\n"
      f"{metrics.classification_report(y_test, predictions)}\n")

# Tree Algorithms

# Non-parametric supervised learning method used for classification or regression. The deeper the tree, the more complex the decision rules and the fitter the model. It is simple to understand and to interpret.

# Main disadvantage: decision-tree learners can create over-complex trees that do not generalize the data well (overfitting). Setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem. Also decision tree learners create biased trees if some classes dominate.

decision_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 40, random_state = 0) # scikit-learn uses an optimized version of the CART algorithm; gini criterion is much faster as it is less expensive to compute, whereas entropy does log calculation and is a more expensive computation (i.e. gini is more efficient in terms if computing power); set max depth to 40 to avoid overfitting

decision_tree.fit(X_train, y_train) # build a decision tree classifier from the training set

predictions = decision_tree.predict(X_test) # predict class for X
print(predictions[0:9])
print(np.transpose(y_test[0:9]))
print("Decision Tree Classifier (CART Algorithm): Training Score {}, Test Score {}".format(decision_tree.score(X_train, y_train), decision_tree.score(X_test, y_test))) # mean accuracy on the given data and labels

_, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (12, 6))
axes = axes.flatten()

for ax, image, prediction in zip(axes, X_test, predictions):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap = plt.cm.gray_r, interpolation = 'nearest')
    ax.set_title(f'Prediction: {prediction}')

plt.savefig(os.path.join(path, 'PredictionsCART.png'))
plt.close()

cm = metrics.confusion_matrix(y_test, predictions)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap='Blues')

plt.savefig(os.path.join(path, 'ConfusionMatrixCART.png'))
plt.close()

print(f"Classification Report {decision_tree} Classifier:\n"
      f"{metrics.classification_report(y_test, predictions)}\n")

# Ensemble Machine Learning 

# Ensemble learning methods are meta-algorithms that combine several machine learning algorithms into a single predictive model to increase performance. Ensemble models offer greater accuracy than individual base classifiers.

# Ensembles simply means combining multiple models.

	# AdaBoost Classifier

# Adaptive Boosting is one of the ensemble boosting classifier. It combines multiple weak classifiers to increase the accuracy of classifiers. 

adaboost = AdaBoostClassifier(algorithm = 'SAMME', learning_rate = 1.0, n_estimators = 100, random_state = 0) # SAMME discrete boosting algorithm; base estimator by default is Decision Tree classifier

adaboost.fit(X_train, y_train) # build a boosted classifier from the training set

predictions = adaboost.predict(X_test) # predict class for X
print(predictions[0:9])
print(np.transpose(y_test[0:9]))
print("Adaboost Classifier: Training Score {}, Test Score {}".format(adaboost.score(X_train, y_train), adaboost.score(X_test, y_test))) # mean accuracy on the given data and labels


_, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (12, 6))
axes = axes.flatten()

for ax, image, prediction in zip(axes, X_test, predictions):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap = plt.cm.gray_r, interpolation = 'nearest')
    ax.set_title(f'Prediction: {prediction}')

plt.savefig(os.path.join(path, 'PredictionsAdaboost.png'))
plt.close()

cm = metrics.confusion_matrix(y_test, predictions)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap='Blues')

plt.savefig(os.path.join(path, 'ConfusionMatrixAdaboost.png'))
plt.close()

print(f"Classification Report {adaboost} Classifier:\n"
      f"{metrics.classification_report(y_test, predictions)}\n")
      
	# Bagging Classifier
	
# An emsemble meta estimator that fits base classifiers each on random subsets of the original dataset and the aggregate their individual predictions to form a final prediction. The common way to combine the predictions is take the majority vote.

# Main idea is train one model on random samples of the training data in an attempt to reduce its variance. It can reduce variance of the predictions without compromising its accuracy.

bagging = BaggingClassifier(estimator = SVC(), n_estimators = 100, max_features = 0.5, random_state = 0) # estimator used C-Support vector classification, base estimator is a Decision Tree Classifier

bagging.fit(X_train, y_train) # build a bagging model ensemble of estimators from training set

predictions = bagging.predict(X_test) # predict class for X
print(predictions[0:9])
print(np.transpose(y_test[0:9]))
print("Bagging Classifier (Estimator: C-Support Vector): Training Score {}, Test Score {}".format(bagging.score(X_train, y_train), bagging.score(X_test, y_test))) # mean accuracy on the given data and labels

bagging = BaggingClassifier(n_estimators = 100, max_features = 0.5, random_state = 0) # estimator used Decision Tree Classifier

bagging.fit(X_train, y_train) # build a bagging model ensemble of estimators from training set

predictions = bagging.predict(X_test) # predict class for X
print(predictions[0:9])
print(np.transpose(y_test[0:9]))
print("Bagging Classifier (Estimator: Decision  Tree Classifier): Training Score {}, Test Score {}".format(bagging.score(X_train, y_train), bagging.score(X_test, y_test))) # mean accuracy on the given data and labels

_, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (12, 6))
axes = axes.flatten()

for ax, image, prediction in zip(axes, X_test, predictions):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap = plt.cm.gray_r, interpolation = 'nearest')
    ax.set_title(f'Prediction: {prediction}')

plt.savefig(os.path.join(path, 'PredictionsBagging.png'))
plt.close()

cm = metrics.confusion_matrix(y_test, predictions)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap='Blues')

plt.savefig(os.path.join(path, 'ConfusionMatrixBagging.png'))
plt.close()

print(f"Classification Report {bagging} Classifier:\n"
      f"{metrics.classification_report(y_test, predictions)}\n")

	# Random Forest Classifier
	
# Ensemble learning method. Random forest is a meta estimator that fits a number of decision tree classifiers in various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

# Random forest randomly selects observations, builds a decision tree, and takes the average result. It doesn't use any set of formulas.

random_forest = RandomForestClassifier(n_estimators = 150, max_depth = None, criterion = 'gini', random_state = 0) # no. of estimators set to 500, usually the more estimators the better it will do, Oshiro et al. (2012) reported that there is no significant improvement after 128 trees; a single decision tree do need pruning in order to overcome over-fitting issue, however, in random forest, this issue is eliminated by random selecting the variables 

random_forest.fit(X_train, y_train) # build a forest of trees from the training set

predictions = random_forest.predict(X_test) # predict class for X
print(predictions[0:9])
print(np.transpose(y_test[0:9]))
print("Random Forest Classifier: Training Score {}, Test Score {}".format(random_forest.score(X_train, y_train), random_forest.score(X_test, y_test))) # mean accuracy on the given data and labels

_, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (12, 6))
axes = axes.flatten()

for ax, image, prediction in zip(axes, X_test, predictions):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap = plt.cm.gray_r, interpolation = 'nearest')
    ax.set_title(f'Prediction: {prediction}')

plt.savefig(os.path.join(path, 'PredictionsRandomForest.png'))
plt.close()

cm = metrics.confusion_matrix(y_test, predictions)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap='Blues')

plt.savefig(os.path.join(path, 'ConfusionMatrixRandomForest.png'))
plt.close()

print(f"Classification Report {random_forest} Classifier:\n"
      f"{metrics.classification_report(y_test, predictions)}\n")
      
# Baseline Model Performance

models = []
models.append(('Log Regression', LogisticRegression(fit_intercept = True, multi_class = 'auto', penalty = 'l2', solver = 'saga', max_iter = 1000, dual = False, random_state = 0)))
models.append(('MLP', MLPClassifier(activation = 'tanh', solver = 'adam', alpha = 0.32, max_iter = 300, random_state = 0)))
models.append(('SVM', SVC(C = 1.0, kernel = 'rbf', decision_function_shape = 'ovo', random_state = 0)))
models.append(('Decision Tree', DecisionTreeClassifier(criterion = 'entropy', max_depth = 40, random_state = 0)))
models.append(('Adaboost', AdaBoostClassifier(algorithm = 'SAMME', learning_rate = 1.0, n_estimators = 100, random_state = 0)))
models.append(('Bagging', BaggingClassifier(estimator = SVC(), n_estimators = 100, max_features = 0.5, random_state = 0)))
models.append(('Random Forest', RandomForestClassifier(n_estimators = 150, max_depth = None, criterion = 'gini', random_state = 0)))

	# Model Accuracy
	
print("Evaluating Model Accuracy.\n")

results = []
names = []
times = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits = 10)
    start_time = time.time()
    cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)
    elapsed_time = (time.time() - start_time) / 10  # training time per fold
    results.append(cv_results * 100)
    names.append(name)
    times.append(elapsed_time)
    msg = "%s: %.2f, Std %.2f, time: %.4f seconds" % (name, cv_results.mean(), cv_results.std(), elapsed_time)
    print(msg)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Baseline Model Performance (Accuracy)')
plt.boxplot(results, notch = False, vert = False, medianprops = dict(color = 'cornflowerblue'))
plt.tick_params(
	axis = 'x',
	which = 'both',
	bottom = False)
plt.tick_params(
	axis = 'y',
	which = 'both',
	left = False
)
ax.set_yticklabels(names)
ax.grid(True, linestyle = '-', linewidth = 1, alpha = 0.3) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for spine in ax.spines.values():
    spine.set_alpha(0.2)
plt.show()

	# Model Precision
	
print("Evaluating Model Precision.\n")

results = []
names = []
times = []
scoring = 'precision'
for name, model in models:
    kfold = KFold(n_splits = 10)
    start_time = time.time()
    cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = f'{scoring}_macro') # if average = 'macro' precision/recall is computed for each class and then the average is taken
    elapsed_time = (time.time() - start_time) / 10  # training time per fold
    results.append(cv_results * 100)
    names.append(name)
    times.append(elapsed_time)
    msg = "%s: %.2f, Std %.2f, time: %.4f seconds" % (name, cv_results.mean(), cv_results.std(), elapsed_time)
    print(msg)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Baseline Model Performance (Precision)')
plt.boxplot(results, notch = False, vert = False, medianprops = dict(color = 'orange'))
plt.tick_params(
	axis = 'x',
	which = 'both',
	bottom = False)
plt.tick_params(
	axis = 'y',
	which = 'both',
	left = False
)
ax.set_yticklabels(names)
ax.grid(True, linestyle = '-', linewidth = 1, alpha = 0.3) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for spine in ax.spines.values():
    spine.set_alpha(0.2)
plt.show()

