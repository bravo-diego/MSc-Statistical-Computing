# Text Clustering using K-Means Algorithm

# Approach for grouping similar documents based on their semantic content. The main objective is to facilitate the organization, summarization, and discovery of patterns in large collections of text data.

import re
import nltk
import fasttext
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim.models import KeyedVectors

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.cluster import hierarchy 
from scipy.cluster.hierarchy import dendrogram, linkage

from fcmeans import FCM

years = list(range(2019, 2024)) # years list from 2019 to 2023
months = list(range(1, 13)) # months list from 1 to 12
days = list(range(1, 32)) # days list from 1 to 31
weekdays = list(range(1, 8)) # days in a week
string_month = 'month' # month as string
files = [] # list to store csv files
week = [] # empty list to store transcriptions per week 
i = 1 # weekday counter

	# Retrieve documents from the given github repository for years 2019 to 2023 

for year in years: 
	for month in months:
		if month == 1:
			string_month = 'enero'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv' # path file
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = [] 
					i = 1 
		elif month == 2:
			string_month = 'febrero'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = [] 
					i = 1 
		elif month == 3:
			string_month = 'marzo'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = [] 
					i = 1 
		elif month == 4:
			string_month = 'abril'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = []
					i = 1 
		elif month == 5:
			string_month = 'mayo'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = [] 
					i = 1 
		elif month == 6:
			string_month = 'junio'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = [] 
					i = 1 
		elif month == 7:
			string_month = 'julio'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) 
					i += 1
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = []
					i = 1 
		elif month == 8:
			string_month = 'agosto'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = [] 
					i = 1 
		elif month == 9:
			string_month = 'septiembre'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = [] 
					i = 1 
		elif month == 10:
			string_month = 'octubre'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = [] 
					i = 1 
		elif month == 11:
			string_month = 'noviembre'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = [] 
					i = 1 
		elif month == 12:
			string_month = 'diciembre'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T4/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					week.append(pd.read_csv(path)) # read csv file
					i += 1 
				except FileNotFoundError:
					i += 1 
					pass # continue if file is not found
				if i == 8:
					files.append(week) # grouping documents per week
					week = [] 
					i = 1 
files.append(week) 
  
# Text Preprocessing 

transcriptions_per_week = [] # list to store text transcriptions grouped by week
for file in files:
	transcriptions = []
	for document in file:
		try:
			text = document['Texto']
			transcriptions.append(text.str.cat(sep = ' ') )
		except FileNotFoundError:
			pass
	transcriptions_per_week.append(transcriptions)

def text_preprocessing(text):
	text = text.lower() # all characters to lowercase 
	text = re.sub(r'[^\w\s]', '', text) # remove special charecters, punctuation and spaces
	text = re.sub(r'\d+', '', text) # remove digits
	text = re.sub(r'\W*\b\w{1,4}\b', '', text) # remove all words that is 3 characters or less
	tokens = word_tokenize(text) # tokenization divides a string into a list of tokens; tokens can be though of as a word in a text
	stop_words = set(stopwords.words('spanish')) # common words in any language that occur with a high frequency but carry much less substantive information about the meaning of a phrase or text
	tokens = [w for w in tokens if w not in stop_words] # remove all stop words
	
	lemmatizer = WordNetLemmatizer()
	tokens = [lemmatizer.lemmatize(token) for token in tokens] # lemmatization; grouping together different inflected forms of a word so they can be analyzed as a single item; it links words with similar meanings to one word

	transcription = ' '.join(tokens)
	
	transcription = re.sub(r'[àáâãäå]', 'a', transcription) # remove common accent characters
	transcription = re.sub(r'[èéêë]', 'e', transcription)
	transcription = re.sub(r'[ìíîï]', 'i', transcription)
	transcription = re.sub(r'[òóôõö]', 'o', transcription)
	transcription = re.sub(r'[ùúûü]', 'u', transcription)
	
	return transcription
	
nltk.download('stopwords') # download stopwords via NLTK data installer
	
processed_transcriptions = [] # empty list to store processed transcriptions
for week in transcriptions_per_week: 
	processed_transcriptions_week = []
	for text in week:
		processed_transcriptions_week.append(text_preprocessing(text)) # apply text preprocessing function to every transcription in the list
	processed_transcriptions.append(processed_transcriptions_week)

documents = 0
for week in processed_transcriptions:
    for text in week:
        documents += 1 # verify no. of transcriptions 

print(documents)
print(len(processed_transcriptions)) # there are 1,206 transcriptions grouped in 266 weeks

corpus = ' '.join(element for slist in processed_transcriptions for element in slist)
corpus = [corpus]

max_words = 2000
vectorizer = TfidfVectorizer(lowercase = False, ngram_range = (1,1), max_features = max_words) # feature extraction method
X = vectorizer.fit_transform(corpus)
vocabulary = vectorizer.get_feature_names_out() # vocabulary

# Word Embeddings

# Word embeddings are basically representations where contexts and similarities are captured by encoding in a vector space similar words would have similar representations; language modeling technique for mapping words to vectors of real numbers 

	# Word2Vec Embedding

path = "/home/aspphem/Desktop/MCE/DataScience/T4/Scripts/sbw_vectors.bin" # Word2Vec trained model
model = KeyedVectors.load_word2vec_format(path, binary = True) # Word2Vec creates a representation of each word present in our vocabulary into a vector

print(model.most_similar("casa")) # verify if model is correct loaded; similarity results for a specific word

vectors = [model[word] if word in model else np.zeros(model.vector_size) for word in vocabulary]
vectors = np.array(vectors)
	
# Clustering Text Documents using K-Means

# K means algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within cluster  sum of squares.

# Getting the Optimal Number of Clusters

# Elbow criterion method - sum of squared errors (SSE) tend to decrease toward 0 as we increase k (SSE is 0 when k is equal to the number of data points in the dataset, because each data point is its own cluster, and there is no error between it and the center of its cluster). The goal is to choose a small value of k that still has a low SSE.

# Silhouette coefficient method - a higher silhouette coefficient score relates to a model with better-defined clusters (i.e. a higher silhouette coefficient indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters). Silhouette coefficient is defined for each sample and is composed of two scores: 1) a as the mean distance between a sample and all other points in the same class; 2) b as the mean distance between a sample and all other points in the next nearest cluster; s = (b - a)/max(a, b) silhouette coefficient for a sigle sample.

range_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
elbow = []
ss = []
 
for clusters in range_clusters: # iterating through cluster sizes
   kmeans = KMeans(n_clusters = clusters, random_state = 0)
   cluster_labels = kmeans.fit_predict(vectors)
   silhouette_avg = silhouette_score(vectors, cluster_labels) # average silhouette score
   ss.append(silhouette_avg)
   print("For clusters = ", clusters,"The average silhouette_score is :", silhouette_avg)
   elbow.append(kmeans.inertia_) # inertia defined as the sum of distances of samples to their closest cluster center
   
fig = plt.figure(figsize = (10, 6))
fig.add_subplot(121)
plt.plot(range_clusters, elbow, 'k-', label = 'Sum of Squared Error', color = 'dimgray')
plt.xlabel("No. Cluster")
plt.ylabel("SSE")
plt.legend()

fig.add_subplot(122)
plt.plot(range_clusters, ss, 'k-', label = 'Silhouette Score', color = 'dimgray')
plt.xlabel("No. Cluster")
plt.ylabel("Silhouette Score")
plt.legend()

plt.show()

clust = AgglomerativeClustering(n_clusters = None, distance_threshold = 0)
clust.fit_predict(vectors)

def plot_dendrogram(model, **kwargs): # create a linkage matrix and plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    hierarchy.set_link_color_palette(['orange', 'cornflowerblue', 'gray', 'black'])
    
    dendrogram(linkage_matrix, color_threshold = None, above_threshold_color='#bcbddc', **kwargs)

plt.figure(figsize = (12, 8))
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(clust)
plt.xlabel("No. points in node")
plt.axhline(y = 14.5, color = 'r')
plt.show()

k = 5 # no. of centroids to generate
kmeans = KMeans(n_clusters = k, random_state = 0, n_init = "auto")
kmeans.fit(vectors)

centroids = kmeans.cluster_centers_ # coordinates of cluster centers
distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis = 2) # distances to each cluster center
normalized_distances = 1 - (distances / np.max(distances, axis = 0)) # normalized distances
membership = pd.DataFrame(normalized_distances, columns = ['Cluster_{}'.format(i + 1) for i in range(k)])
membership.index = vocabulary

for i in range(k):
    topic_weights = dict(zip(vocabulary, membership['Cluster_{}'.format(i + 1)]))
    wordcloud = WordCloud(width = 800, height = 400, background_color = 'white').generate_from_frequencies(topic_weights) # word clouds
    plt.figure(figsize = (8, 6))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.title('Topic no. {}'.format(i + 1))
    plt.axis('off')
    plt.show()

labels = kmeans.labels_
colors = ['orange', 'cornflowerblue', 'slategray', 'khaki', 'indianred']

pca = PCA(n_components = 2) # principal component analysis
topic_assignment_pca = pca.fit_transform(vectors) # fit the model with x and apply the dimensionality reduction on x

pca_df = pd.DataFrame(topic_assignment_pca, columns = ['PC1', 'PC2'])
pca_df['Cluster'] = labels

plt.figure(figsize = (8, 6))
for i in range(k):
    cluster_data = pca_df[pca_df['Cluster'] == i]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label = f'Topic {i+1}', color = colors[i], alpha = 0.6)

plt.title('Clusters Visualization 2D Space (PCA)\nWord2Vec Model')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(False)
plt.show()

kpca = KernelPCA(n_components = 2, kernel = 'rbf') # kernel principal component analysis 
topic_assignment_kpca = kpca.fit_transform(vectors) # fit the model from data in x and transform x

kpca_df = pd.DataFrame(topic_assignment_kpca, columns  =  ['Component 1', 'Component 2'])
kpca_df['Cluster'] = labels

plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_data = kpca_df[kpca_df['Cluster'] == i]
    plt.scatter(cluster_data['Component 1'], cluster_data['Component 2'], label = f'Topic {i+1}', color = colors[i], alpha = 0.6)

plt.title('Clusters Visualization 2D Space (Kernel-PCA)\nWord2Vec Model')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(False)
plt.show()

tsne = TSNE(n_jobs = -1, random_state = 0) # t-distributed stochastic neighbor embedding
topic_assignment_tsne = tsne.fit_transform(vectors) # fit x into an embedded space and return that transformed output 

tsne_df = pd.DataFrame(topic_assignment_tsne, columns = ['x', 'y'])
tsne_df['Cluster'] = labels

plt.figure(figsize = (8, 6))
for i in range(k):
    cluster_data = tsne_df[tsne_df['Cluster'] == i]
    plt.scatter(cluster_data['x'], cluster_data['y'], label = f'Topic {i+1}', color = colors[i], alpha = 0.6)

plt.title('Clusters Visualization 2D Space (tSNE)\nWord2Vec Model')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(False)
plt.show()

	# FastText 

# Library for efficient learning of word representations and sentence classification

path = "/home/aspphem/Desktop/MCE/DataScience/T4/Scripts/embeddings-l-model.bin" # Word2Vec trained model
model = fasttext.load_model(path) # FastText object

print(model.words) # verify if model is correct loaded; similarity results for a specific word

vectors = {word: model.get_word_vector(word) for word in vocabulary}
vectors = np.array(list(vectors.values()))

k = 5 # no. of centroids to generate
kmeans = KMeans(n_clusters = k, random_state = 0, n_init = "auto")
kmeans.fit(vectors)

centroids = kmeans.cluster_centers_ # coordinates of cluster centers
distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis = 2) # distances to each cluster center
normalized_distances = 1 - (distances / np.max(distances, axis = 0)) # normalized distances
membership = pd.DataFrame(normalized_distances, columns = ['Cluster_{}'.format(i + 1) for i in range(k)])
membership.index = vocabulary

for i in range(k): 
    topic_weights = dict(zip(vocabulary, membership['Cluster_{}'.format(i + 1)]))
    wordcloud = WordCloud(width = 800, height = 400, background_color = 'white').generate_from_frequencies(topic_weights) # word clouds
    plt.figure(figsize = (8, 6))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.title('Topic no. {}'.format(i + 1))
    plt.axis('off')
    plt.show()

labels = kmeans.labels_
colors = ['orange', 'cornflowerblue', 'slategray', 'khaki', 'indianred']

pca = PCA(n_components = 2) # principal component analysis
topic_assignment_pca = pca.fit_transform(vectors) # fit the model with x and apply the dimensionality reduction on x

pca_df = pd.DataFrame(topic_assignment_pca, columns = ['PC1', 'PC2'])
pca_df['Cluster'] = labels

plt.figure(figsize = (8, 6))
for i in range(k):
    cluster_data = pca_df[pca_df['Cluster'] == i]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label = f'Topic {i+1}', color = colors[i], alpha = 0.6)

plt.title('Clusters Visualization 2D Space (PCA)\nFastText Model')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(False)
plt.show()

kpca = KernelPCA(n_components = 2, kernel = 'rbf') # kernel principal component analysis 
topic_assignment_kpca = kpca.fit_transform(vectors) # fit the model from data in x and transform x

kpca_df = pd.DataFrame(topic_assignment_kpca, columns  =  ['Component 1', 'Component 2'])
kpca_df['Cluster'] = labels

plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_data = kpca_df[kpca_df['Cluster'] == i]
    plt.scatter(cluster_data['Component 1'], cluster_data['Component 2'], label = f'Topic {i+1}', color = colors[i], alpha = 0.6)

plt.title('Clusters Visualization 2D Space (Kernel-PCA)\nFastText Model')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(False)
plt.show()

tsne = TSNE(n_jobs = -1, random_state = 0) # t-distributed stochastic neighbor embedding
topic_assignment_tsne = tsne.fit_transform(vectors) # fit x into an embedded space and return that transformed output 

tsne_df = pd.DataFrame(topic_assignment_tsne, columns = ['x', 'y'])
tsne_df['Cluster'] = labels

plt.figure(figsize = (8, 6))
for i in range(k):
    cluster_data = tsne_df[tsne_df['Cluster'] == i]
    plt.scatter(cluster_data['x'], cluster_data['y'], label = f'Topic {i+1}', color = colors[i], alpha = 0.6)

plt.title('Clusters Visualization 2D Space (tSNE)\nFastText Model')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(False)
plt.show()

	# Note: transcriptions and word embeddings were retrieved and downloaded from this github repositories: 1) https://github.com/NOSTRODATA/conferencias_matutinas_amlo 2) https://github.com/dccuchile/spanish-word-embeddings 

