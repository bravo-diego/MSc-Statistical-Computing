# Principal Component Analysis (PCA)

# Linear dimensionality reduction technique. It transforms a set of correlated variables (p) into a smaller k (k < p) number of uncorrelated variables (principal components) while retaining as much of the variation in the original dataset as possible. PCA takes advantage of existing correlations between the input variables in the dataset and combines those correlated variables into a new set of uncorrelated variables. Is a unsupervised machine learning algorithm as it doesn't requiere labels in the data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from matplotlib.patches import Circle
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage, TextArea)

lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 1) # loading LFW dataset; it's specified to include only images of people who have at least 70 images in the dataset, and the images are displayed in the original size (125 x 94)

for name in lfw_people.target_names:
    print(name)

n_samples, h, w = lfw_people.images.shape

X = lfw_people.data # each row corresponds to a ravelled face image of original; numpy array

print(X.shape[0])
print(X.shape[1])

n_features = X.shape[1]

y = lfw_people.target # labels associated to each face image; numpy array
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, target_names[y], train_size = 0.80, test_size = 0.20, random_state = 42) # split data into training and testing sets; where 80% of the data is used for training and the remaining 20% is used for testing

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

no_components = 50 # no. of principal components to be extracted from the data; principal components are the orthogonal vectors that provide a lower-dimensional representation of the data while preserving the maximum amount of variance in the original dataset (i.e. no. components determines how many of these principal components we want to retain, typically is set to a value less than or equal to the original number of features in the dataset). Selecting to few components may result in loss of important information, while selecting too many may lead to overfitting or unnecessarily high-dimensional data

pca = PCA(n_components = no_components, svd_solver = 'randomized', whiten = True).fit(X_train) # compute a PCA (eigenfaces) on the face dataset

eigenfaces = pca.components_.reshape((no_components, h, w)) # extract eigenfaces (principal components)

X_train_pca = pca.transform(X_train) # X is projected on the first principal components previously extracted from a training set; i.e. projection of X in the first principal components, where n_samples is the number of samples and n_components is the number of the components

explained_variance = pca.explained_variance_ratio_ # eigenvalues (explained variance); percentage of variance explained by each of the selected components

print(explained_variance * 100) # one-dimensional numpy array which contains the values of the percentage of variance explained by each of the selected components

print(np.cumsum(explained_variance * 100)) # the first 50 principal components keep about 80% of the variability in the dataset 

# Scree plots are used to visualize the eigenvalues of the principal components; the eigenvalues represent the amount of variance explained by each principal component. 

pca_values = np.arange(pca.n_components) + 1 
plt.plot(pca_values, explained_variance, '-', linewidth = 2, color = 'steelblue')
plt.title('Scree Plot')
plt.xlabel('Principal Component') # number of principal components on the x-axis
plt.ylabel('Explained Variance') # eigenvalues on the y-axis
for pos in ['right', 'top']: # spines visibility as False
	plt.gca().spines[pos].set_visible(False)
plt.show() # the scree plot helps to identify the number of principal components that capture the most variance in the data

plt.figure(figsize=(8, 6)) # PCA loading plot of the first two principal components
for i in range(len(target_names)):
    plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], label=target_names[i])

plt.title('Principal Component Analysis LFW Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Score Plot\nVariables Projected onto PC1 and PC2')
plt.legend(loc='best')
plt.grid(False)
for pos in ['right', 'top']: # spines visibility as False
	plt.gca().spines[pos].set_visible(False)
plt.show()

pca_df = pd.DataFrame(data=X_train_pca[:, :2], columns=['Principal Component 1', 'Principal Component 2'])
images = [] # empty list to store images
for i in range(len(X_train)):
	images.append(X_train[i].reshape((h, w))) 

print(len(images), len(pca_df['Principal Component 1']), len(pca_df['Principal Component 2']))

fig, ax = plt.subplots() # PCA loading plot of the first two principal components plotting face images instead points in order to identify possible patterns

for x, y, image in zip(pca_df['Principal Component 1'], pca_df['Principal Component 2'], images):
	imagebox = OffsetImage(image, zoom=0.2, cmap=plt.cm.gray)	
	ab = AnnotationBbox(imagebox, (x, y), xybox=(30., -30.), xycoords='data', boxcoords='offset points', frameon = False)
	ax.add_artist(ab)

ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Score Plot\nVariables Projected onto PC1 and PC2')
plt.tick_params(left = True, right = False, labelleft = True, labelbottom = True, bottom = True)
for pos in ['right', 'top']: # spines visibility as False
	plt.gca().spines[pos].set_visible(False)
fig.set_size_inches(12, 12)
plt.show()

X_test_pca = pca.transform(X_test) # projection of test data in the first principal components

clf = RandomForestClassifier(n_estimators = 90)
clf.fit(X_train_pca, y_train) # train a classifier on PCA features
y_pred = clf.predict(X_test_pca)

def plot_gallery(images, titles, h, w, n_row = 3, n_col = 4): # function to plot a gallery of portraits
    plt.figure(figsize = (1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom = 0, left = 0.01, right = 0.99, top = 0.90, hspace = 0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap = plt.cm.gray)
        plt.title(titles[i], size = 12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1] # predicted names
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1] # true names
    return 'Predicted: %s\n True: %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w) # plot the result of the prediction on a portion of the test set

plt.show()

# Nearest Neighbors

# The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points. The distance can, in general, be any metric measure, standard euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods, since they simply 'remember' all of its training data.

neigh = NearestNeighbors(n_neighbors = 5, metric = 'euclidean') 
neigh.fit(X_train_pca) # fit the nearest neighbors estimator from the training set; returns the fitted nearest neighbors estimator

distances, indices = neigh.kneighbors(X_train_pca[50].reshape(1, -1)) # find the k-neighbors of a point; returns indices of and distances to the neighbors of each point
index_nearest_neighbor = indices[0][1] # nearest neighbor indices 
nn_image = X_train[index_nearest_neighbor].reshape(h, w) # nearest neighbor image

train_image = X_train[50].reshape(h, w) # image from the training data
projected_train_image = pca.inverse_transform(X_train_pca[50]).reshape(h, w) # transform data back to its original space; X_train_pca contains the training data projected onto the principal components, it has shape (1030, 25), indicating that has 50 elements, not enough to reshape into an image with dimensions (h = 125, w = 94). Thus we can't directly visualize a single row as an image because it represents the coefficients of that sample along the principal components, NOT the actual image data

def plot_images(images, titles, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

images = [train_image, projected_train_image, nn_image]
titles = ["Test Image", "Projection Image", "Nearest Neighbor Image"]

plot_images(images, titles)



