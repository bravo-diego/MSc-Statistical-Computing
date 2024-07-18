# Feature Importance 

# Refers to techniques that assign a score to input features based on how useful they are at predicting a target variable. 

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier

X_train = pd.read_csv('/home/aspphem/Desktop/MCE/DataScience/T5/SuppMaterial/mnist/mnist_Xtrain.csv', header = None).to_numpy() # training data set 
y_train = pd.read_csv('/home/aspphem/Desktop/MCE/DataScience/T5/SuppMaterial/mnist/mnist_Ytrain.csv', header = None).to_numpy() # set of labels to all the data in train set

X_test = pd.read_csv('/home/aspphem/Desktop/MCE/DataScience/T5/SuppMaterial/mnist/mnist_Xtest.csv', header = None).to_numpy() # test data set
y_test = pd.read_csv('/home/aspphem/Desktop/MCE/DataScience/T5/SuppMaterial/mnist/mnist_Ytest.csv', header = None).to_numpy() # set of labels to all the data in test set

print(X_train[0].shape)

X_train = X_train[0:1000,:]
y_train = y_train[0:1000,:]

X_test = X_test[0:200,:]
y_test = y_test[0:200,:]

y = y_train.ravel()
y_train = np.array(y).astype(int)

def feature_importance(importance, features_names, title, no_features = 20): # feature importances; feature names; plot title; no. of features to be plotted 
    indices = np.argsort(importance)[-no_features:]

    plt.figure(figsize = (8, 6))
    #plt.title(title)
    plt.barh(range(len(indices)), importance[indices], color = 'lightsteelblue', align = 'center')
    plt.yticks(range(len(indices)), [features_names[i] for i in indices])
    plt.tick_params(axis = 'x', which = 'both', top = False)
    plt.tick_params(axis = 'y', which = 'both', right = False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tick_params(axis = 'y', which = 'both', left = False)
    plt.xlabel('Feature Importance')
    plt.show()

feature_names = [f'Pixel {i}' for i in range(X_train.shape[1])] # feature names 

decision_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 40, random_state = 0)
decision_tree.fit(X_train, y_train) # classification model need to be fitted in order to compute the feature importances

feature_importance(decision_tree.feature_importances_, feature_names, 'Feature Importance CART Classifier') # apply feature_importance function; feature importances provided by attribute feature_importances_

adaboost = AdaBoostClassifier(algorithm = 'SAMME', learning_rate = 1.0, n_estimators = 100, random_state = 0)
adaboost.fit(X_train, y_train) 

feature_importance(adaboost.feature_importances_, feature_names, 'Feature Importance AdaBoost Classifier')

random_forest = RandomForestClassifier(n_estimators = 150, max_depth = None, criterion = 'gini', random_state = 0) 
random_forest.fit(X_train, y_train)

feature_importance(random_forest.feature_importances_, feature_names, 'Feature Importance Random Forest Classifier')






