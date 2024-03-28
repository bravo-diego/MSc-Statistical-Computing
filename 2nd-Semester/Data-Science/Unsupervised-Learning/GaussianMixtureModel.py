# Gaussian Mixture Model

# A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. Mixture models can be viewed as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples = 1000, centers = 5, cluster_std = 0.60, random_state = 0)
X = StandardScaler().fit_transform(X)

plt.figure(figsize=(8, 6)) # synthetic data

plt.scatter(X[:, 0], X[:, 1], c = 'slategray', alpha = 0.5) 
plt.title('Synthetic Data')
plt.grid(False)
plt.show()

clusters = 5 # no. of clusters

gaussian_mixture = GaussianMixture(n_components = clusters, random_state = 0) #  GaussianMixture object implements the expectation-maximization (EM) algorithm for fitting mixture-of-Gaussian models
gaussian_mixture.fit(X)
gaussian_mixture_labels = gaussian_mixture.predict(X) # predicted labels

# Fuzzy k-means 

# The clusters produced by the k-means procedure are sometimes called hard or crisp clusters, since any feature vector x either is or not a member of a particular cluster. This is in contrast to soft or fuzzy clusters, in which a feature vector x can have a degree of membership in each cluster.

	# Make initial guesses for the means \mu_{i} for i = (1, 2, ..., k)

	# Until there are no changes in any mean 

		# Use the estimated means to find the degree of membership u(j,i) of x_{j} in Cluster i

		# For i from 1 to k 
	
			# Replace \mu_{i} with the fuzzy mean of all of the examples for Cluster i 

		# end for

	# end until

# No o t e: in the fuzzy-k-means procedure, the degree of membership can also be interpreted probabilistically as the square root of the a posteriori probability the x is in Cluster i.

center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, clusters, 2, error = 0.005, maxiter = 1000, init = None) # fuzzy k-means
fuzzy_kmeans_labels = np.argmax(u, axis=0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c = gaussian_mixture_labels, cmap = 'magma', alpha = 0.5)
plt.title('Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c = fuzzy_kmeans_labels, cmap = 'magma', alpha = 0.5)
plt.title('Fuzzy K-Means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()


