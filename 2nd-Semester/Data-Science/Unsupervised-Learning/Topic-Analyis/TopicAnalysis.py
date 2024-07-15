# Topic Analysis

# TF-IDF is a combination of two different words (term frequency and inverse documents frequency). Term frequency (TF) is used to measure that how many times a term is present in a document. Inverse document frequency (IDF) assigns weights to the words given their frequency in a document, it assigns lower weight to frequent words and assings greater weight for the words that are infrequent. Thus TF-IDF is a numerical statistic that shows the relevance of keywords to some specific documents or it can be said that, it provides those keywords, using which some specific documents can be identified or categorized.

# "TF-IDF is nothing, but just the multiplication of term frequency (TF) and inverse document frequency (IDF)..." Qaiser S. & Ali R. (2018). Text Mining: Use of TF-IDF to Examine the Relevance of Words to Documents.

import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

years = list(range(2019, 2024)) # years list from 2019 to 2023
months = list(range(1, 13)) # months list from 1 to 12
days = list(range(1, 32)) # days list from 1 to 31
string_month = 'month' # month as string
files = [] # list to store csv files
 
	# Retrieve documents from the given github repository for years 2019 to 2023
 
for year in years: 
	for month in months:
		if month == 1:
			string_month = 'enero'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv' # path file
				try:
					files.append(pd.read_csv(path)) # read csv file
				except FileNotFoundError:
					pass # continue if file is not found
				
		elif month == 2:
			string_month = 'febrero'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					pass 
		elif month == 3:
			string_month = 'marzo'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					pass 
		elif month == 4:
			string_month = 'abril'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					pass 
		elif month == 5:
			string_month = 'mayo'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					continue 
		elif month == 6:
			string_month = 'junio'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:from years 2019 to 2023
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					pass 
		elif month == 7:
			string_month = 'julio'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					pass 
		elif month == 8:
			string_month = 'agosto'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					pass 
		elif month == 9:
			string_month = 'septiembre'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					pass 
		elif month == 10:
			string_month = 'octubre'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					pass 
		elif month == 11:
			string_month = 'noviembre'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					pass 
		elif month == 12:
			string_month = 'diciembre'
			for day in days:
				path = f'/home/aspphem/Desktop/MCE/DataScience/T3/SuppMaterial/conferencias_matutinas_amlo-master/{year}/{month}-{year}/{string_month} {day}, {year}/csv_por_participante/PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR.csv'
				try:
					files.append(pd.read_csv(path))
				except FileNotFoundError:
					pass 
from years 2019 to 2023
print(len(files)) # there are 1,206 csv files (transcriptions)
print(files[0]['Texto'].str.cat(sep = ' ')) # transcription no. 1; concatenate sfrom years 2019 to 2023trings in the series/index with given separator, separator between different elements given by a space

# Text Preprocessing 

transcriptions = [] # list to store text transcriptions

for file in range(len(files)):
	try:
		transcription = files[file]['Texto'].str.cat(sep = ' ')
		transcriptions.append(transcription)
	except FileNotFoundError:
		pass
	
print(len(transcriptions)) # there are 1,206 transcriptions

def text_preprocessing(text):
	text = text.lower() # all characters to lowercase 
	text = re.sub(r'[^\w\s]', '', text) # remove special charecters, punctuation and spaces
	text = re.sub(r'\d+', '', text) # remove digits
	text = re.sub(r'\W*\b\w{1,3}\b', '', text) # remove all words that is 3 characters or less
	text = re.sub(r'[àáâãäå]', 'a', text) # remove common accent characters
	text = re.sub(r'[èéêë]', 'e', text)
	text = re.sub(r'[ìíîï]', 'i', text)
	text = re.sub(r'[òóôõö]', 'o', text)
	text = re.sub(r'[ùúûü]', 'u', text)

	tokens = word_tokenize(text) # tokenization divides a string into a list of tokens; tokens can be though of as a word in a text
	stop_words = set(stopwords.words('spanish')) # common words in any language that occur with a high frequency but carry much less substantive information about the meaning of a phrase or text
	tokens = [w for w in tokens if w not in stop_words] # remove all stop words
	
	lemmatizer = WordNetLemmatizer()
	tokens = [lemmatizer.lemmatize(token) for token in tokens] # lemmatization; grouping together different inflected forms of a word so they can be analyzed as a single item; it links words with similar meanings to one word
	
	transcription = ' '.join(tokens)
	
	return transcription
	
processed_transcriptions = []
for transcription in transcriptions:
	processed_transcriptions.append(text_preprocessing(transcription))

print(len(processed_transcriptions)) # there are 1,206 transcriptions
print(processed_transcriptions[0])

# Term Frequency - Inverse Document Frequency (TF-IDF)

no_words = 250

	# Count vectorizer: frequency of words in a document

vectorizer = CountVectorizer(lowercase = False, ngram_range = (1,1), max_features = no_words, binary = False) # convert a collection of text documents to a matrix of token counts; 
X = vectorizer.fit_transform(processed_transcriptions) # learn a vocabulary dictionary of all tokens in the raw documents
token_counts = X.toarray()
token_counts = pd.DataFrame(token_counts, columns = vectorizer.get_feature_names_out())
print(token_counts) # frequency of each word in the document; each row represents a document and each column represents a unique word in the corpus

	# TF-IDF vectorizer: importance of words in a document

tfidf_vectorizer = TfidfVectorizer(lowercase = False, ngram_range = (1,1), max_features = no_words, binary = False) # convert a collection of raw documents to a matrix of TF-IDF features
X = tfidf_vectorizer.fit_transform(processed_transcriptions) # learn vocabulary 
tfidf_features = X.toarray()
tfidf_features = pd.DataFrame(tfidf_features, columns = tfidf_vectorizer.get_feature_names_out())
print(tfidf_features) # frequency of each word in the document; each row represents a document and each column represents a unique word in the corpus

SVD = TruncatedSVD(n_components = 100) # dimensionality reduction using truncated SVD 

tfidf_transformed = SVD.fit_transform(tfidf_features) # fit the model to X and perform dimensionality reduction on X
print(tfidf_transformed)

feature_names = tfidf_vectorizer.get_feature_names_out() # retrieve feature names from TF-IDF vectorizer
topics = pd.DataFrame(SVD.components_, columns = feature_names) # each row corresponds to a topic and each column corresponds to a word in a vocabulary; values are weights of each word in each topic, obtained from the SVD components

k = 8 # no. of topics

for i in range(k):
	weights = dict(zip(feature_names, topics.iloc[i])) # feature names (words) as keys; weights of words as values
	wordcloud = WordCloud(width = 1800, height = 1000, background_color = 'white').generate_from_frequencies(weights) # word cloud generator
	plt.figure(figsize = (6, 4))
	plt.imshow(wordcloud, interpolation = 'bilinear')
	plt.axis('off')
	plt.show()

index = np.argmax(tfidf_transformed, axis = 1) + 1 # assigns every transcription to a topic 

pca = PCA() # principal component analysis; linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
topic_pca = pca.fit_transform(tfidf_transformed) # linear dimensionality reduction through PCA

kernel_pca = KernelPCA(n_components = 8, kernel = 'rbf') # kernel principal component analysis; non-linear dimensionality reduction through the use of kernels
topic_kernel_pca = kernel_pca.fit_transform(tfidf_transformed) # linear dimensionality reduction through Kernel-PCA
	
t_SNE = TSNE(n_components = 2) # T-distributed stochastich neighbor embedding
topic_tsne = t_SNE.fit_transform(tfidf_transformed) # linear dimensionality reduction through t-SNE

plt.figure(figsize=(15, 5))  

plt.subplot(131)
plt.scatter(topic_pca[:, 0], topic_pca[:, 1], c=index, cmap='cividis')
plt.title('PCA')

plt.subplot(132)
plt.scatter(topic_kernel_pca[:, 0], topic_kernel_pca[:, 1], c=index, cmap='cividis')
plt.title('Kernel PCA')

plt.subplot(133)
plt.scatter(topic_tsne[:, 0], topic_tsne[:, 1], c=index, cmap='cividis')
plt.title('t-SNE')

plt.tight_layout()  
plt.show()

# Non-negative matrix factorization

# Finds two non-negative matrices (i.e. matrices with all non-negative elements (W, H) whose product approximates the non-negative matrix X). This factorization can be used for dimensionality reduction or topic extraction

nmf = NMF(n_components = 10, max_iter = 400) # non-negative matrix factorization
nmf_transformed = nmf.fit_transform(tfidf_features) # learn a nmf model for the data and returns the transformed data

index = np.argmax(nmf_transformed, axis = 1) # assigns every transcription to a topic 

feature_names = tfidf_vectorizer.get_feature_names_out() # retrieve feature names from TF-IDF vectorizer
topics_nmf = pd.DataFrame(nmf.components_, columns = feature_names) # each row corresponds to a topic and each column corresponds to a word in a vocabulary; values are weights of each word in each topic, obtained from the SVD components

k = 6 # no. of topics

for i in range(k):
	weights = dict(zip(feature_names, topics_nmf.iloc[i])) # feature names (words) as keys; weights of words as values
	wordcloud = WordCloud(width = 1800, height = 1000, background_color = 'white').generate_from_frequencies(weights) # word cloud generator
	plt.figure(figsize = (6, 4))
	plt.imshow(wordcloud, interpolation = 'bilinear')
	plt.axis('off')
	plt.show()
	
pca = PCA() # principal component analysis; linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space
topic_pca = pca.fit_transform(nmf_transformed ) # linear dimensionality reduction through PCA

kernel_pca = KernelPCA(n_components = 20, kernel = 'rbf') # kernel principal component analysis; non-linear dimensionality reduction through the use of kernels
topic_kernel_pca = kernel_pca.fit_transform(nmf_transformed ) # linear dimensionality reduction through Kernel-PCA
	
t_SNE = TSNE(n_components = 2) # T-distributed stochastich neighbor embedding
topic_tsne = t_SNE.fit_transform(nmf_transformed ) # linear dimensionality reduction through t-SNE

plt.figure(figsize=(15, 5))  

plt.subplot(131)
plt.scatter(topic_pca[:, 0], topic_pca[:, 1], c=index, cmap='cividis')
plt.title('PCA')

plt.subplot(132)
plt.scatter(topic_kernel_pca[:, 0], topic_kernel_pca[:, 1], c=index, cmap='cividis')
plt.title('Kernel PCA')

plt.subplot(133)
plt.scatter(topic_tsne[:, 0], topic_tsne[:, 1], c=index, cmap='cividis')
plt.title('t-SNE')

plt.tight_layout()  
plt.show()
	
	# Note: transcriptions were retrieved from this github repository https://github.com/NOSTRODATA/conferencias_matutinas_amlo

