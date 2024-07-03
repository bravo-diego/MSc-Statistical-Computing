# The file 'IMM_2020.xls' corresponds to the marginalization index data for each municipality in the country, proposed by the Consejo Nacional de Poblacion (CONAPO) in 2020, and is the most recent. The file also contains the values of different indicators that represent nine forms of exclusion from marginalization in the dimensions of education, housing, population distribution, and monetary income. These indicators were constructed based on information from the 2020 Population census performed by INEGI. These indicators are used to calculate the marginalization index as described in the technical-methodological note.

# Perform a Principal Component Analysis (PCA) based on the 9 CONAPO indicators. What can you say about the phenomenon of marginalization in the country based on this analysis? Do you find any interesting patterns regarding the municipalities? Construct an alternative marginalization index using the first, second, and third principal component obtained with PCA.

import os
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from openpyxl.workbook import Workbook

path = '/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/IMM_2020.xls' # file path
df = pd.read_excel(path, sheet_name = "IMM_2020") # reading .xls file

print(df.head(), df.shape) # show up first 5 rows  
print(df.keys()) # column names

cols = ['ANALF', 'SBASC', 'OVSDE', 'OVSEE', 'OVSAE', 'OVPT', 'VHAC', 'PL.5000', 'PO2SM'] # socioeconomic indicators; marginalization index according to INEGI 2020

updated_df = df[cols] # remove unnecessary columns; use [] notation to keep columns of interest

X = StandardScaler().fit_transform(updated_df) # standardize data; data standardization is the process of converting data to a common format to enable users to process and analyze it 
cov = np.cov(X.T) # estimate a covariance matrix

eigenvalues, eigenvectors = np.linalg.eig(cov) # eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1] # sort eigenvalues
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

var_explained = {'PC':range(1,10), 'std':np.sqrt(eigenvalues), 'var_prop':eigenvalues/sum(eigenvalues), 'cum_prop':np.cumsum(eigenvalues/sum(eigenvalues))} # exlained variance; measure of how much of the total variance in the original dataset is explained by each principal component
stds = pd.DataFrame(data = var_explained)

print(stds) # the first 6 principal components keep about 90% of the variability in the dataset 

plt.figure(figsize = (8, 6))
sns.lineplot(x = 'PC', y = 'var_prop', data = stds) 
plt.xlabel('Principal Component') # number of principal components on the x-axis
plt.ylabel('Explained Variance') # eigenvalues on the y-axis
plt.title('Scree Plot')
for pos in ['right', 'top']: # spines visibility as False
	plt.gca().spines[pos].set_visible(False)
plt.show() # the scree plot helps to identify the number of principal components that capture the most variance in the data

components = pd.DataFrame(data = eigenvectors.T, columns = updated_df.columns, index  = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9'])
print(components)

X = pd.DataFrame(X) 
pca = PCA()
pca.fit(X)

components = pd.DataFrame(data = pca.components_.T, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9'], index = updated_df.columns) # principal components data frame using attribute pca.components (principal axes in feature space, representing the directions of maximum variance in the data)
print(components)

projection = pd.DataFrame(pca.transform(X), columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9']) # projection in the principal components

plt.figure(figsize = (8, 6))
plt.scatter(projection['PC1'], projection['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Score Plot\nVariables Projected onto PC1 and PC2')
for pos in ['right', 'top']: # spines visibility as False
	plt.gca().spines[pos].set_visible(False)
plt.grid(False)
plt.show()

plt.figure(figsize = (8, 6))
plt.scatter(projection['PC1'], projection['PC3'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 3')
plt.title('Score Plot\nVariables Projected onto PC1 and PC3')
for pos in ['right', 'top']: # spines visibility as False
	plt.gca().spines[pos].set_visible(False)
plt.grid(False)
plt.show()

PCA = PCA(n_components = 9)
components = PCA.fit_transform(X)
PCA.components_

def biplot(score, coeff, labels = None): # function to create a PCA biplot
    x = score[:, 0] # principal component 1
    y = score[:, 1] # principal component 2
    n = coeff.shape[0]
    scale_x = 1.0/(x.max() - x.min())
    scale_y = 1.0/(y.max() - y.min())
    plt.figure(figsize = (10, 8))
    plt.scatter(x * scale_x, y * scale_y, s = 5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color = 'crimson', alpha = 0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var"+str(i+1), color = 'lightgreen', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0] * 1.15, coeff[i,1] * 1.15, labels[i], color = 'seagreen', ha = 'center', va = 'center')
    for pos in ['right', 'top']: # spines visibility as False
        plt.gca().spines[pos].set_visible(False)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Biplot of PCA')
    plt.grid(False)
    plt.show()

components = PCA.fit_transform(X)
biplot(components, np.transpose(PCA.components_), list(updated_df.columns))

# Marginalization Index Map by CONAPO 2020

file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/national/national_estatal.shp') # reading cartography files
base = gpd.read_file(file)
file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/mg2021_integrado_tarea/conjunto_de_datos/00mun.shp')
layer = gpd.read_file(file, index_col = 'CVEGE0')
layer = layer.to_crs("EPSG:4326") 

file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/IMM_2020.xls') # marginalization index data 
lst_str_cols = ['CVE_ENT', 'NOM_ENT', 'CVE_MUN', 'NOM_MUN'] # states codes as strings
dict_dtypes = {x : 'str'  for x in lst_str_cols}
marg_municipal = pd.read_excel(file, sheet_name = 'IMM_2020', dtype = dict_dtypes) # sheet name in xls file
marg_municipal = marg_municipal.set_index('CVE_MUN') 
layer_marg = layer.merge(marg_municipal, left_on = 'CVEGEO', right_on = 'CVE_MUN')
im_cat = pd.CategoricalIndex(layer_marg['GM_2020'], categories = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']) # GM_2020 column as categorical variable 

f, ax = plt.subplots(1, figsize = (15, 15))
ax.set_aspect('equal')
base.plot(color = 'white', edgecolor = 'black', ax = ax) 
layer_marg.plot(column = 'GM_2020', categorical = True, legend = True, linewidth = 0, ax = ax, cmap = 'Reds', categories = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']) 
ax.set_axis_off() # remove axis
plt.show()

# Marginalization Index Map usign Principal Components 

	# IMEyM Based on 1st Principal Component

IM_df = df.copy()
IM_df['PC1'] = projection['PC1'] # add PC1 column to df copy

scaler = MinMaxScaler()
IM_df['PC1'] = scaler.fit_transform(IM_df[['PC1']]) # fit and transform the PC1 column
print(IM_df['PC1'])

IM_df.to_excel('IMM_2020_PC1.xlsx', sheet_name = 'IMM_2020', index = False) # Note: just run this line once to create a new xls file with PC1 column

file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/national/national_estatal.shp') # reading cartography files
base = gpd.read_file(file)
file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/mg2021_integrado_tarea/conjunto_de_datos/00mun.shp')
layer = gpd.read_file(file, index_col = 'CVEGE0')
layer = layer.to_crs("EPSG:4326") 
 
file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/IMM_2020_PC1.xls') # marginalizaition index data
lst_str_cols = ['CVE_ENT', 'NOM_ENT', 'CVE_MUN', 'NOM_MUN'] # states codes as strings
dict_dtypes = {x : 'str'  for x in lst_str_cols}
marg_municipal = pd.read_excel(file, sheet_name = 'IMM_2020', dtype = dict_dtypes) # sheet name in xls file
marg_municipal = marg_municipal.set_index('CVE_MUN') 
layer_marg = layer.merge(marg_municipal, left_on = 'CVEGEO', right_on = 'CVE_MUN')
im_cat = pd.CategoricalIndex(layer_marg['GM_2020'], categories = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']) # GM_2020 column as categorical variable 

f, ax = plt.subplots(1, figsize = (15, 15))
ax.set_aspect('equal')
base.plot(color = 'white', edgecolor = 'black', ax = ax) 
layer_marg.plot(column = 'IMN_2020', categorical = False, legend = True, linewidth = 0, ax = ax, cmap = 'Reds') 
ax.set_axis_off() # remove axis
plt.show()

	# # IMEyM Based on 2nd Principal Component

IM_df = df.copy()
IM_df['PC2'] = projection['PC2'] # add PC2 column to df copy

scaler = MinMaxScaler()
IM_df['PC2'] = scaler.fit_transform(IM_df[['PC2']]) # fit and transform the PC1 column
print(IM_df['PC2'])

IM_df.to_excel('IMM_2020_PC2.xlsx', sheet_name = 'IMM_2020', index = False) # Note: just run this line once to create a new xls file with PC2 column

file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/national/national_estatal.shp') # reading cartography files
base = gpd.read_file(file)
file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/mg2021_integrado_tarea/conjunto_de_datos/00mun.shp')
layer = gpd.read_file(file, index_col = 'CVEGE0')
layer = layer.to_crs("EPSG:4326") 
 
file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/IMM_2020_PC2.xls') # marginalizaition index data
lst_str_cols = ['CVE_ENT', 'NOM_ENT', 'CVE_MUN', 'NOM_MUN'] # states codes as strings
dict_dtypes = {x : 'str'  for x in lst_str_cols}
marg_municipal = pd.read_excel(file, sheet_name = 'IMM_2020', dtype = dict_dtypes) # sheet name in xls file
marg_municipal = marg_municipal.set_index('CVE_MUN') 
layer_marg = layer.merge(marg_municipal, left_on = 'CVEGEO', right_on = 'CVE_MUN')
im_cat = pd.CategoricalIndex(layer_marg['GM_2020'], categories = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']) # GM_2020 column as categorical variable 

f, ax = plt.subplots(1, figsize = (15, 15))
ax.set_aspect('equal')
base.plot(color = 'white', edgecolor = 'black', ax = ax) 
layer_marg.plot(column = 'IMN_2020', categorical = False, legend = True, linewidth = 0, ax = ax, cmap = 'Reds') 
ax.set_axis_off() # remove axis
plt.show()

	# # IMEyM Based on 3rd Principal Component

IM_df = df.copy()
IM_df['PC3'] = projection['PC3'] # add PC3 column to df copy

scaler = MinMaxScaler()
IM_df['PC3'] = scaler.fit_transform(IM_df[['PC3']]) # fit and transform the PC1 column
print(IM_df['PC3'])

IM_df.to_excel('IMM_2020_PC3.xlsx', sheet_name = 'IMM_2020', index = False) # Note: just run this line once to create a new xls file with PC3 column

file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/national/national_estatal.shp') # reading cartography files
base = gpd.read_file(file)
file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/mg2021_integrado_tarea/conjunto_de_datos/00mun.shp')
layer = gpd.read_file(file, index_col = 'CVEGE0')
layer = layer.to_crs("EPSG:4326") 
 
file = os.path.join('/home/aspphem/Desktop/MCE/DataScience/T1/SuppMaterial/IMM_2020_PC3.xls') # marginalizaition index data
lst_str_cols = ['CVE_ENT', 'NOM_ENT', 'CVE_MUN', 'NOM_MUN'] # states codes as strings
dict_dtypes = {x : 'str'  for x in lst_str_cols}
marg_municipal = pd.read_excel(file, sheet_name = 'IMM_2020', dtype = dict_dtypes) # sheet name in xls file
marg_municipal = marg_municipal.set_index('CVE_MUN') 
layer_marg = layer.merge(marg_municipal, left_on = 'CVEGEO', right_on = 'CVE_MUN')
im_cat = pd.CategoricalIndex(layer_marg['GM_2020'], categories = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']) # GM_2020 column as categorical variable 

f, ax = plt.subplots(1, figsize = (15, 15))
ax.set_aspect('equal')
base.plot(color = 'white', edgecolor = 'black', ax = ax) 
layer_marg.plot(column = 'IMN_2020', categorical = False, legend = True, linewidth = 0, ax = ax, cmap = 'Reds') 
ax.set_axis_off() # remove axis
plt.show()

