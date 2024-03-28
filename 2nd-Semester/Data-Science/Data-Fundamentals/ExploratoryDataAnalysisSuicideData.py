# Given the data in the 'suicide_data.csv' file, which compiles information related to suicides in 101 countries over different years and information related to country development indicators, such as Gross Domestic Product per capita and Human Development Index; perform an exploratory and descriptive data analysis using Pandas.

	# Importing the necessary libraries 

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from string import ascii_letters

	# Loading the Dataset	

path = "/home/aspphem/Desktop/MCE/DataScience/T1/Datasets/suicide_data.csv"
df = pd.read_csv(path)

	# Exploring and Summarizing Data

print(df.head()) # show up first 5 rows  
print(df.tail(), df.shape) # show up last 5 rows; (27820 rows, 12 columns)
print(df.keys()) # column names

print(df.describe(include = 'object')) # data summary
print(df.isna().sum()) # there are missing values in 'HDI for year' column
print("Duplicated values: {}".format(df.duplicated().sum())) # there are not duplicated values

print(df.iloc[:, 3].unique()) # unique age intervals
print(df.iloc[:, 0].unique()) # unique country names

	# Preprocessing the Data

df.rename(columns={"country": "Country", "year": "Year", "sex": "Sex", "age": "Age_Interval", "suicides_no": "No_Suicides", "population": "Population", "suicides/100k pop": "Suicides_100KPop", "country-year": "Country_Year", "HDI for year": "HDI_Year", " gdp_for_year ($) ": "GDP_Year", "gdp_per_capita ($)": "GDP_Capita", "generation": "Generation"}, inplace = True) # rename columns names

print(df.head()) # updated column names

# Handle missing values

not_null_df = pd.notnull(df['HDI_Year'])
print(df[not_null_df].shape) # no. rows having 'HDI_Year = Value' displayed

null_df = pd.isnull(df['HDI_Year'])
print(df[null_df].shape) # no. rows having 'HDI_Year = NaN' displayed

updated_df = df.fillna(method = 'bfill') # bfill() method replaces missing values with the values from the next row

cols = ['Country', 'Year', 'Sex', 'Age_Interval', 'Population', 'No_Suicides', 'Suicides_100KPop', 'HDI_Year', 'GDP_Capita'] # columns of interest 

updated_df = updated_df[cols] # remove unnecessary columns; use [] notation to keep columns of interest
print(updated_df.head(), updated_df.shape) # (27820 rows, 9 columns)

	# Exploratory Data Analysis

# General look about the data

df.hist(figsize=(12,8))
plt.show()

# Correlation matrix

latam_countries = ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Mexico', 'Uruguay'] # countries of interest

latam_df = updated_df.loc[updated_df['Country'].isin(latam_countries)]

print(latam_df.iloc[:, [4, 5, 6, 7, 8]].corr()) # correlation matrix using data frame with filling NaN values using 'bfill' method

mask = np.triu(np.ones_like(latam_df.iloc[:, [4, 5, 6, 7, 8]].corr(), dtype = bool))
f, ax = plt.subplots(figsize = (9.5, 7.5)) # figure size
cmap = sns.diverging_palette(230, 20, as_cmap = True)

sns.heatmap(latam_df.iloc[:, [4, 5, 6, 7, 8]].corr(), mask = mask, cmap = cmap, vmax = 0.3, center = 0, square = True, linewidths = 0.5, cbar_kws = {"shrink": 0.5}) # plot diagonal correlation matrix

plt.show()

# Global suicide rates

female_df = updated_df[updated_df['Sex'] == 'female'] # split dataframe in two subgroups; female dataframe
male_df = updated_df[updated_df['Sex'] == 'male'] # male dataframe

no_suicides_female = female_df['No_Suicides'].sum() # total no. of female suicides
no_suicides_male = male_df['No_Suicides'].sum() # total no. of male suicides

fig1, ax1 = plt.subplots(figsize=(6, 4)) # figure size
ax1.pie([no_suicides_female, no_suicides_male], explode = (0.1, 0), labels = ['Female', 'Male'], colors = ['#FC8D62', '#8DA0CB'], autopct = '%1.1f%%', shadow = True, startangle = 90) # pie chart; explode specifies the fraction of the radius with which to offset each wedge; autopct is a string or function used to label the wedges with their numeric value; startangle specifies the angle by which the start of the pie is rotated (counterclockwise from the x-axis)
ax1.axis('equal') # pie drawn as a circle

plt.title('Global Suicide Rates by Gender')
plt.show() # male suicide rates are observed to be generally higher than female rates across the analyzed data

# Number of suicides in Latin America countries by gender

female_grouped_df = female_df.groupby(['Country', 'Year']).agg({'No_Suicides': 'sum', 'HDI_Year': 'first', 'GDP_Capita': 'first'}).reset_index() # no. of female suicides grouped by both country and year 

male_grouped_df = male_df.groupby(['Country', 'Year']).agg({'No_Suicides': 'sum', 'HDI_Year': 'first', 'GDP_Capita': 'first'}).reset_index() # no. of male suicides grouped by both country and year 

plt.figure(figsize=(12, 10)) # figure size
plt.suptitle("Number of Suicides Over Time by Gender", fontsize="x-large") # add global title

countries = ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Mexico', 'Paraguay']  # countries of interest

for i, country in enumerate(countries, start=1):
    plt.subplot(3, 2, i)

    country_data_female = female_grouped_df[female_grouped_df['Country'] == country] # females plot
    plt.plot(country_data_female['Year'], country_data_female['No_Suicides'], '#FC8D62', label='Female')

    country_data_male = male_grouped_df[male_grouped_df['Country'] == country] # males plot
    plt.plot(country_data_male['Year'], country_data_male['No_Suicides'], '#8DA0CB', label='Male')

    plt.title(country)
    
    if i % 2 != 0:
    	plt.ylabel('No. of Suicides') 
    if i > 4:
    	plt.xlabel('Year')
    
    for pos in ['right', 'top']: # spines visibility as False
        plt.gca().spines[pos].set_visible(False)
	
    plt.grid(False)

handles, labels = plt.gca().get_legend_handles_labels() 
plt.figlegend(handles, labels, loc='upper right') # add a global legend
plt.tight_layout()
plt.show() # general tendency remains consistent, no. of male suicides are higher than no. of female suicides, both globally and in latin america countries

# Number of suicides and GDP per Capita over time  

grouped_df = updated_df.groupby(['Country', 'Year']).agg({'No_Suicides': 'sum', 'HDI_Year': 'first', 'GDP_Capita': 'first'}).reset_index() # both female and male no. of suicides

print(grouped_df.head())

countries = ['Brazil', 'Mexico', 'Argentina', 'Colombia', 'Chile', 'Uruguay'] # countries for plotting

palette = ['#B30000', '#E34A33', '#FC8D59', '#FDBB84', '#FDD49E', '#FEF0D9'] # custom color palette no. of suicides plot
sns.palplot(sns.color_palette(palette))

plt.figure(figsize=(8.5, 4.2))  # plotting the no. of suicides over time by country

for country, color in zip(countries, palette):
    country_data = grouped_df[grouped_df['Country'] == country]
    plt.plot(country_data['Year'], country_data['No_Suicides'], label=country, color=color)

plt.title('Number of Suicides Over Time by Country')
plt.xlabel('Year')
plt.ylabel('Number of Suicides')
plt.legend()
plt.grid(False)
for pos in ['right', 'top']: # spines visibility as False
	plt.gca().spines[pos].set_visible(False)
plt.show() # number of suicides remains increasing over time across latin america countries 

countries = ['Uruguay', 'Chile', 'Argentina', 'Brazil', 'Mexico', 'Colombia'] # countries for plotting

palette = ['#0868AC', '#43A2CA', '#7BCCC4', '#A8DDB5', '#CCEBC5', '#F0F9E8'] # custom color palette gpd per capita plot
sns.palplot(sns.color_palette(palette))

plt.figure(figsize=(8.5, 4.2)) # plotting the gpd per capita over time by country

for country, color in zip(countries, palette):
    country_data = grouped_df[grouped_df['Country'] == country]
    plt.plot(country_data['Year'], country_data['GDP_Capita'], label=country, color=color)

plt.title('GDP per Capita Over Time by Country')
plt.xlabel('Year')
plt.ylabel('GDP per Capita ($)')
plt.legend()
plt.grid(False)
plt.show()

# Note that although the plots provide a general overview of the trend in suicide rates for men and women in Latin America, additional information are still needed to fully understands the factors that may explains this phenomenon.


