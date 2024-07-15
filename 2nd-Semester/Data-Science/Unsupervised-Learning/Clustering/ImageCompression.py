# Clustering algorithms are used to split a dataset into several groups (i.e. clusters), so that the objects in the same group are as similar as possible and the objects in different groups are as dissimilar as possible.

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from io import BytesIO
from sklearn import preprocessing
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA

try:
	img_original = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po.png") # an object Image type is returned and stored in img variable
except IOError: # raise an 'IOError' if file cannot be found or image cannot be opened
	pass

def imageSize(img_original):
	img_file = BytesIO()
	image = Image.fromaray(np.uint8(img_original))
	image.save(img_file, 'png')
	return img_file.tell()/1024

img_size = os.path.getsize("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po.png") # original image size
img_original = img_original.convert('RGB')	
	
img_original = np.asarray(img_original)
print(img_original[0:2], img_original.shape) # RGB pixel-values; dimensions of the image
	
plt.imshow(img_original)
plt.axis('off')  
plt.show() # show picture
	
pixels = img_original.reshape(img_original.shape[0] * img_original.shape[1], img_original.shape[2])
print(pixels.shape) # change shape of the picture with img_array.shape[0] * img_array.shape[1] rows and img_array.shape[2] columns to feed the data in the algorithm

	# K-means Clustering 

# Unsupervised learning algorithm that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid). The main objective of this algorithm is to minimize the sum of distances between the data point and their corresponding clusters.

# The algorithm takes the unlabeled dataset as input, divides the dataset into k-number of clusters, and repeats the process until it doesn't find the best clusters. Note: the value of k should be predetermined in this algorithm.

no_clusters = 5 # no. of clusters to form as well as the number of centroids to generate; i.e. we want to keep just 'no_clusters' colours in the compressed image, the lesser the no. of cluster the more compressed will the image be, hence the quality of the image will also decrease
model = KMeans(n_clusters = no_clusters )
model.fit(pixels) # compute k-means clustering; 'pixels' as training instances to cluster 

centroids = model.labels_
cluster_centers = model.cluster_centers_

print(centroids.shape) # cluster number that is assigned to each data point or each pixel
print(cluster_centers.shape) # coordinates or the RGB values of the cluster centers

x = np.zeros((centroids.shape[0], 3))	
print(x.shape)

for i in range(no_clusters): # assign the cluster centers to the data points corresponding to the cluster to which they belong
	x[centroids == i] = cluster_centers[i] # iterate through all the clusters and assing the cluster centroids (RGB values) to each data point, reducing the image into 16 colours

compressed_image = x.reshape(img_original.shape[0], img_original.shape[1], 3)
compressed_image = Image.fromarray(np.uint8(compressed_image))
compressed_image.save('Po_Compressed_K5.png')

try:
	img_compressed = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Po_Compressed_K5.png") # an object Image type is returned and stored in img variable
except IOError: # raise an 'IOError' if file cannot be found or image cannot be opened
	pass
	
def imageSize(img_compressed):
	img_file = BytesIO()
	image = Image.fromaray(np.uint8(img_compressed))
	image.save(img_file, 'png')
	return img_file.tell()/1024

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1) # 1 row, 2 columns; 1st image
plt.imshow(img_original)
plt.title('Original Image') # original image
plt.axis('off')

plt.subplot(1, 2, 2) # 2nd image
plt.imshow(img_compressed)
plt.title('Compressed Image\nK-means Clustering') # compressed k-means clustering image
plt.axis('off')

plt.show() 

	 # Principal Component Analysis (PCA)
 
 # Linear dimensionality reduction technique. It transforms a set of correlated variables (p) into a smaller k (k < p) number of uncorrelated variables (principal components) while retaining as much of the variation in the original dataset as possible. PCA takes advantage of existing correlations between the input variables in the dataset and combines those correlated variables into a new set of uncorrelated variables. Is a unsupervised machine learning algorithm as it doesn't requiere labels in the data. Reduces the no. of dimensions in large datasets to principal components that retain most of the original information.

img_original_normalized = ((img_original.T - img_original.mean((1,2))) / img_original.std((1,2))).T # normalization of RGB image; normalization means to transform to zero mean and unit variance

r, g, b = cv2.split(img_original) # split channels of coloured image

print(r, r.shape)
print(g, g.shape)
print(b, b.shape)

no_components = 5 # no. of principal components to be extracted from the data; principal components are the orthogonal vectors that provide a lower-dimensional representation of the data while preserving the maximum amount of variance in the original dataset 

pca = PCA(n_components = no_components, svd_solver = 'randomized', whiten = True).fit(r) # compute a PCA on each channel 
r_pca = pca.transform(r) # X is projected on the first principal components previously extracted from a training set; i.e. projection of X in the first principal components, where n_samples is the number of samples and n_components is the number of the components
print(r_pca.shape)
r_pca_inverse = pca.inverse_transform(r_pca).reshape(img_original.shape[0], img_original.shape[1]) # transform data back to its original space
print(r_pca_inverse.shape)

pca = PCA(n_components = no_components, svd_solver = 'randomized', whiten = True).fit(g) 
g_pca = pca.transform(g)
print(g_pca.shape)
g_pca_inverse = pca.inverse_transform(g_pca).reshape(img_original.shape[0], img_original.shape[1]) 
print(g_pca_inverse.shape)

pca = PCA(n_components = no_components, svd_solver = 'randomized', whiten = True).fit(b) 
b_pca = pca.transform(b)
print(b_pca.shape)
b_pca_inverse = pca.inverse_transform(b_pca).reshape(img_original.shape[0], img_original.shape[1]) 
print(b_pca_inverse.shape)

img_compressed = cv2.merge((r_pca_inverse, g_pca_inverse, b_pca_inverse)) 
print(img_compressed.shape)
 
compressed_image = Image.fromarray(np.uint8(img_compressed))
compressed_image.save('Po_Compressed_p5.png')

try:
	img_compressed = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Po_Compressed_p5.png") # an object Image type is returned and stored in img variable
except IOError: # raise an 'IOError' if file cannot be found or image cannot be opened
	pass
	
def imageSize(img_compressed):
	img_file = BytesIO()
	image = Image.fromaray(np.uint8(img_compressed))
	image.save(img_file, 'png')
	return img_file.tell()/1024

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1) # 1 row, 2 columns; 1st image
plt.imshow(img_original)
plt.title('Original Image') # original image
plt.axis('off')

plt.subplot(1, 2, 2) # 2nd image
plt.imshow(img_compressed)
plt.title('Compressed Image\nPrincipal Component Analysis') # compressed pca image
plt.axis('off')

plt.show() 

# Comparison between images modifying the number of clusters (K-means approach) and principal components (PCA approach) 

plt.figure(figsize=(10, 5)) # K-means Clustering Approach

plt.subplot(2, 2, 1) # 3 row, 2 columns; 1st image
img = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po.png")
plt.imshow(img)
plt.title('Original Image\nImage Size: {0:.2f} Kb'.format(img_size/1000)) # original image
plt.axis('off')


plt.subplot(2, 2, 2) # 2nd image
img = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_K5.png")
plt.imshow(img)
plt.title('Compressed Image (K = 5)\nImage Size: {0:.2f} Kb'.format(os.path.getsize("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_K5.png")/1000)) # compressed k-means clustering image
plt.axis('off')

plt.subplot(2, 2, 3) # 3rd image
img = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_K15.png")
plt.imshow(img)
plt.title('Compressed Image (K = 15)\nImage Size: {0:.2f} Kb'.format(os.path.getsize("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_K15.png")/1000)) # compressed k-means clustering image
plt.axis('off')

plt.subplot(2, 2, 4) # 4th image
img = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_K25.png")
plt.imshow(img)
plt.title('Compressed Image (K = 25)\nImage Size: {0:.2f} Kb'.format(os.path.getsize("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_K25.png")/1000)) # compressed k-means clustering image
plt.axis('off')

plt.show() 

plt.figure(figsize=(10, 5)) # Principal Component Analysis Approach

plt.subplot(2, 2, 1) # 3 row, 2 columns; 1st image
img = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po.png")
plt.imshow(img)
plt.title('Original Image\nImage Size: {0:.2f} Kb'.format(img_size/1000)) # original image
plt.axis('off')


plt.subplot(2, 2, 2) # 2nd image
img = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_p5.png")
plt.imshow(img)
plt.title('Compressed Image (p = 5)\nImage Size: {0:.2f} Kb'.format(os.path.getsize("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_p5.png")/1000)) # compressed pca image
plt.axis('off')

plt.subplot(2, 2, 3) # 3rd image
img = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_p25.png")
plt.imshow(img)
plt.title('Compressed Image (p = 25)\nImage Size: {0:.2f} Kb'.format(os.path.getsize("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_p25.png")/1000)) # compressed pca image
plt.axis('off')

plt.subplot(2, 2, 4) # 4th image
img = Image.open("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_p50.png")
plt.imshow(img)
plt.title('Compressed Image (p = 50)\nImage Size: {0:.2f} Kb'.format(os.path.getsize("/home/aspphem/Desktop/MCE/DataScience/T2/Scripts/Figures/Po_Compressed_p50.png")/1000)) # compressed pca image
plt.axis('off')

plt.show() 


