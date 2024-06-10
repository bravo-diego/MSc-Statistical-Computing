# Analisis de Datos Categoricos para Estadisticas Oficiales: Encuesta Nacional de Calidad e Impacto Gubernamental 2017

# Centro de Investigación en Matemáticas - Maestría en Cómputo Estadístico

import io
import requests
import squarify
import geopandas

import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from dbfread import DBF
from sklearn.impute import KNNImputer
from ydata_profiling import ProfileReport
from IPython.display import FileLink, display

import streamlit as st

alt.data_transformers.enable("vegafusion") # habilita el transformador de datos 'vegafusion' para trabajar con conjuntos de datos >5000 filas

st.set_page_config(page_title = 'ENCIG-2017', page_icon = 'Active', layout = 'wide') # streamlit configuracion
path = 'streamlit/CIMAT.png'
st.image(path, width=100) # logo CIMAT

st.title("Encuesta Nacional de Calidad e Impacto Gubernamental (ENCIG) 2017") # streamlit titulo
st.subheader("Centro de Investigación en Matemáticas, A.C.") # streamlit subtitulo
st.markdown('<style>div.block-container{padding-top:1rem}</style>',unsafe_allow_html = True)
tabs = st.tabs(['Principales Resultados'])

# Cargar conjuntos de datos

def read_dbf_file(path):
	try:
		table = DBF(path, encoding = 'latin1', load = True) # leer archivo .dbf
		df = pd.DataFrame(iter(table)) # convertir tabla a dataframe
		print(df.keys()) # nombres de las columnas
		print(df.shape) # dimensiones del dataframe (filas, columnas)
		return df
	except Exception as e:
		print(f"An error occurred: {e}")

# Tabla encig2017_01_sec1_3_4_5_8_9_10 contiene informacion sobre los residentes de la vivienda y la identificacion de los hogares; percepcion de corrupcion; evaluacion de servicios basicos; evaluacion de servicios publicos bajo demanda; corrupcion y corrupcion general.

# Tabla encig2017_04_sec_7 contiene informacion relacionada con la calidad de los tramites y de los servicios publicos realizados de manera personal por el informante seleccionado.

# Tabla encig2017_01_sec_11 contiene informacion relacionada con la confianza en las instituciones y actores diversos.

# Procesamiento del conjunto de datos

# Tabla del hogar principal y de características del elegido: secciones 1, 3, 4, 5, 8, 9, y 10.

sec_1_3_4_5_8_9_10 = pd.read_csv('streamlit/sec_1_3_4_5_8_9_10.csv')

# Tabla de confianza en las instituciones:  sección 11

sec_11 = pd.read_csv('streamlit/sec_11.csv')

# Tabla de seguimiento de trámites, pagos o servicios públicos: sección 7

sec_7 = pd.read_csv('streamlit/sec_7.csv')

# Principales Problemáticas de Cada Entidad Federativa en la República Mexicana

columns = list(range(20, 32)) 

updated_names = ['Desempeno_Gub', 'Pobreza', 'Corrupcion', 'Desempleo', 'Inseguridad', 'Aplicacion_Ley', 'Desastres_Naturales', 'Educacion', 'Salud', 'Coordinacion_Gub', 'Rendicion_Cuentas', 'Ninguno', 'Entidad_Federativa']

country_main_issues = sec_1_3_4_5_8_9_10.iloc[:, columns].astype(int).copy() # conversion de string a int
country_main_issues['NOM_ENT'] = sec_1_3_4_5_8_9_10['NOM_ENT'] # agregar columna con los nombres de las entidades federativas actualizados
country_main_issues.columns = updated_names # actualizar el nombre de las variables

contingency_table_main_issues = country_main_issues.groupby('Entidad_Federativa').sum()

	# Percepción de la Ocurrencia de las Principales Problemáticas en la República Mexicana

main_issues = ['Desempeño Gubernamental', 'Pobreza', 'Corrupción', 'Desempleo', 'Inseguridad', 'Mala Aplicación de la Ley', 'Medio Ambiente', 'Educación', 'Salud', 'Coordinación Gubernamental', 'Rendición de Cuentas', 'Ninguno']

frecuency_main_issues = contingency_table_main_issues.copy()
frecuency_main_issues.columns = main_issues
frecuency_main_issues = frecuency_main_issues.reset_index()

frecuency_main_issues = pd.melt(frecuency_main_issues, id_vars=['Entidad_Federativa'], value_vars = main_issues, var_name = 'Ploblematica', value_name = 'Frecuencia')

frecuency_by_states = frecuency_main_issues.drop(['Entidad_Federativa'], axis = 1).groupby(['Ploblematica']).sum().reset_index().copy()

	# Distribución Visual de las Principales Problemáticas en la República Mexicana

st.write("# Percepción de la Ocurrencia de las Principales Problemáticas en la República Mexicana") 

col1, col2 = st.columns((2))
with  col1:
	st.markdown(
        """
        <div style='text-align: center; font-size: 36px;'>
        
        
        
            El treemap muestra cómo se distribuyen las principales problemáticas entre la población, destacando la gravedad de cada problemática en términos de su frecuencia reportada. Los dos problemas principales que afectan a la poblacion son en temas de seguridad, con una frecuencia de 27,869, y corrupción, con 21,520 casos. Estas problemáticas están representadas como las áreas más grandes en el treemap.
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
	fig = plt.gcf()
	ax = fig.add_subplot()
	fig.set_size_inches(16, 8)
	squarify.plot(sizes = frecuency_by_states['Frecuencia'], label = frecuency_by_states['Ploblematica'], color = sns.color_palette('PuBu_r', len(frecuency_by_states['Frecuencia'])), alpha = .7) # treemap generado utilizando la libreria squarify

	plt.axis('off')
	st.pyplot(fig)

# Percepción de la Corrupción

columns = list(range(32, 54)) # columnas de interes: variables relacionadas a la percepcion de la corrupcion

updated_names = ['Percepcion', 'Universidades', 'Policia', 'Hospitales', 'Secretarias', 'Empresarios', 'Gubernatura', 'Trabajo', 'Municipio', 'Familiares', 'Sindicatos', 'Vecinos', 'Diputados_Senadores', 'Medios_Comunicacion', 'Intitutos_Electorales', 'Comisiones_Derechos', 'Escuelas_Publicas', 'Jueces_Magistrados', 'Insituciones_Religiosas', 'Partidos_Politicos', 'Ejercito_Marina', 'Ministerio_Publico', 'Entidad_Federativa'] # redefinir nombre de las variables de acuerdo al diccionario de datos de la base de datos

corruption = sec_1_3_4_5_8_9_10 .iloc[:, columns].astype(int).copy() # conversion de string a int

corruption['NOM_ENT'] = sec_1_3_4_5_8_9_10 ['NOM_ENT'] # agregar columna con los nombres de las entidades federativas actualizados
corruption.columns = updated_names # actualizar el nombre de las variables

corruption_perception = corruption.loc[:, ['Entidad_Federativa', 'Percepcion']].copy() # percepcion de la corrupcion por entidad federativa
corruption_perception['Percepcion'] = corruption_perception['Percepcion'].replace({1: 'Muy_Frecuente', 2: 'Frecuente', 3: 'Poco_Frecuente', 4: 'Nada_Frecuente', 9: 'No_Sabe'}) # reemplazar el valor numerico por la categoria correspondiente

contingency_table_corruption_perception = corruption_perception.pivot_table(index = 'Entidad_Federativa', columns = 'Percepcion', aggfunc = len, fill_value = 0) # agrupar observaciones por estado y contar no. de ocurrencias de cada nivel de percepcion

average_level_corruption = contingency_table_corruption_perception.drop(['No_Sabe'], axis = 1).copy() # eliminar variable No Sabe/Responde

average_level_corruption['Presencia'] = average_level_corruption['Muy_Frecuente'] + average_level_corruption['Frecuente'] # variable Presencia incluye observaciones de las categorias Muy Frecuente y Frecuente
average_level_corruption['Ausencia'] = average_level_corruption['Poco_Frecuente'] + average_level_corruption['Nada_Frecuente'] # variable Ausencia incluye observaciones de las categorias Poco Frecuente y Nada Frecuente
average_level_corruption['Total'] = average_level_corruption['Ausencia'] + average_level_corruption['Presencia'] # no. total de observaciones 
average_level_corruption['Porcentaje'] = ((average_level_corruption['Presencia'] / average_level_corruption['Total'])*100).round(2) # porcentaje del nivel de percepcion de corrupcion por entidad federativa

average_level_corruption = average_level_corruption.iloc[:, 4:]

barplot_corruption = average_level_corruption.reset_index().rename(columns = {'Entidad_Federativa': 'Entidad Federativa'}).copy()

	# Panorama General de la Percepción de la Corrupción en México

geographical_data_mx = geopandas.read_file("https://gist.githubusercontent.com/walkerke/76cb8cc5f949432f9555/raw/363c297ce82a4dcb9bdf003d82aa4f64bc695cf1/mx.geojson")
geographical_data_mx.head() # conjunto de datos geográficos de México

geographical_data_mx['state'] = geographical_data_mx['state'].replace({'Aguascalientes': 'AGS', 'Baja California': 'BC', 'Baja California Sur': 'BCS', 'Campeche': 'CAMP', 'Coahuila de Zaragoza': 'COAH', 'Colima': 'COL', 'Chiapas': 'CHIS', 'Chihuahua': 'CHIH', 'Ciudad de México': 'CDMX', 'Durango': 'DGO', 'Guanajuato': 'GTO', 'Guerrero': 'GRO', 'Hidalgo': 'HGO', 'Jalisco': 'JAL', 'México': 'MEX', 'Michoacán de Ocampo': 'MICH', 'Morelos': 'MOR', 'Nayarit': 'NAY', 'Nuevo León': 'NL', 'Oaxaca': 'OAX', 'Puebla': 'PUE', 'Querétaro': 'QRO', 'Quintana Roo': 'QR', 'San Luis Potosí': 'SLP', 'Sinaloa': 'SIN', 'Sonora': 'SON', 'Tabasco': 'TAB', 'Tamaulipas': 'TAM', 'Tlaxcala': 'TLAX', 'Veracruz de Ignacio de la Llave': 'VER', 'Yucatán': 'YUC', 'Zacatecas': 'ZAC'}) # abreviación de los nombres de las entidades federativas para garantizar compatibilidad entre los conjuntos de datos

corruption_data = average_level_corruption.reset_index().copy()
corruption_data = corruption_data.loc[:, ['Entidad_Federativa', 'Porcentaje']] # columnas de interes: entidades federativas y porcentaje de percepción de corrupción por entidad federativa 
names = ['state', 'percent'] 
corruption_data.columns = names 

corruption_map = pd.merge(geographical_data_mx, corruption_data, how = 'inner', on = ['state']) # unir conjunto de datos geograficos con el conjunto de datos de corrupcion apartir de la columna compartida 'state'

st.write("## Nivel de Percepción de la Corrupción por Entidad Federativa") 

#st.subheader("Nivel de Percepción de la Corrupción y Satisfacción General con los Servicios Básicos por Entidad Federativa")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
corruption_map.plot(column = 'percent', cmap = 'OrRd', legend = True, legend_kwds = {"orientation": "horizontal"}, ax = ax)
ax.set_axis_off()

col1, col2 = st.columns((2))
with col1:
	st.pyplot(fig) # percepción de la corrupción

with col2:
	fig = px.bar(barplot_corruption, x = 'Entidad Federativa', y = 'Porcentaje')
	fig.update_layout(xaxis = {'categoryorder':'total descending'}) 
	fig.update_traces(marker_color = 'brown', marker_line_color = 'brown', marker_line_width = 1.5, opacity = 0.6)
	st.plotly_chart(fig) # nivel de percepcion de corrupcion

# Evaluación de Servicios Básicos: Nivel de Satisfacción Ciudadana

basic_services = sec_1_3_4_5_8_9_10.loc[:, ['NOM_ENT', 'P4_1B', 'P4_2B', 'P4_5B', 'P4_6B', 
                                            'P4_7B', 'P5_7B', 'P5_2B', 'P5_4B', 'P5_5B', 'P5_6B']].copy() # columnas de interes correspondientes a la evaluacion de los servicios basicos: agua, policia, luz, transporte publico, salud (IMMS e ISSTE)

services = ['NOM_ENT', 'Agua', 'Drenaje', 'Recoleccion Basura', 'Policia', 'Calles y Avenidas', 'Luz', 'Educacion', 'IMSS', 'ISSTE', 'Seguro Popular'] # servicios publicos basicos
basic_services.columns = services # redefinir las columnas para mejor la legibilidad de los graficos

	#  Imputación de Datos Utilizando el Enfoque K-Vecinos Más Cercanos

basic_services_imputed = basic_services.iloc[:, 1:7].copy() # filtrar servicios 
basic_services_imputed.replace('', np.nan, inplace = True) # reemplazar valores vacios con valores NaN

basic_services_imputed = basic_services_imputed.apply(pd.to_numeric, errors = 'coerce') # convertir valores NaN a valores np.nan para poder trabajar con la funcion KNN Imputer

imputer = KNNImputer(n_neighbors = 2, weights = 'uniform') 
basic_services_updated = pd.DataFrame(imputer.fit_transform(basic_services_imputed), columns = basic_services_imputed.columns)

basic_services_updated = basic_services_updated.astype(int) # conversion de datos tipo float a int
basic_services_updated['NOM_ENT'] = basic_services['NOM_ENT'] # anadir columna con los nombres de las entidades federativas correspondientes a cada observacion

	# Nivel de Satisfacción General con los Servicios Básicos por Entidad Federativa

service_quality_by_state = basic_services_updated.groupby(['NOM_ENT']).sum().copy() # suma de todas las calificaciones por servicio para cada uno de las entidades federativas

n = basic_services_updated['NOM_ENT'].value_counts() # no. total de observaciones por estado

average_service_quality = service_quality_by_state.div(n, axis = 0).round(4).copy() # dividir entradas por el no. total de observaciones para obtener la calificacion promedio de cada servicio para cada entidad federativa
average_service_quality['Calificacion General'] = (average_service_quality.mean(axis = 1)*10).round(4) # calificacion total por entidad federativa basandose en los puntajes obtenidos de cada uno de sus servicios 

barplot_services = average_service_quality.reset_index().rename(columns = {'NOM_ENT': 'Entidad Federativa'}).copy() 

	# Distribución Geográfica de la Satisfacción Percibida con los Servicios Básicos en la República Mexicana

services_data = average_service_quality.reset_index().copy()
services_data = services_data.loc[:, ['NOM_ENT', 'Calificacion General']] # columnas de interes: entidades federativas y calificacion general de los servicios por entidad federativa 
names = ['state', 'percent'] 
services_data.columns = names 

services_satisfaction_map = pd.merge(geographical_data_mx, services_data, how = 'inner', on = ['state']) # unir conjunto de datos geograficos con el conjunto de datos de la calidad de los servicios basicos apartir de la columna compartida 'state'

services_satisfaction_map['percent'] = services_satisfaction_map['percent'].astype(float)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
services_satisfaction_map.plot(column = 'percent', cmap = 'PuBu', legend = True, legend_kwds = {"orientation": "horizontal"}, ax = ax)
ax.set_axis_off()

st.write("## Nivel de Satisfacción General con los Servicios Básicos por Entidad Federativa") 

col1, col2 = st.columns((2))
with col1:
	st.pyplot(fig) # satisfacción general con los servicios básicos
	
with col2:
	fig = px.bar(barplot_services, x = 'Entidad Federativa', y = 'Calificacion General')
	fig.update_layout(xaxis = {'categoryorder':'total descending'}) 
	fig.update_traces(marker_color = 'steelblue', marker_line_color = 'slategray', marker_line_width = 1.5, opacity = 0.6)
	st.plotly_chart(fig) # satisfacción general con los servicios básicos
	
# Nivel de Percepción de la Corrupción y Satisfacción General con los Servicios Básicos por Entidad Federativa
        
barplot_corruption_services = average_level_corruption.copy()
barplot_corruption_services['Calificacion General'] = (average_service_quality['Calificacion General'])
barplot_corruption_services = barplot_corruption_services.sort_values('Porcentaje', ascending = False).reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(
    x = barplot_corruption_services['Entidad_Federativa'],
    y = barplot_corruption_services['Porcentaje'],
    name = 'Percepción de Corrupción',
    marker_color = 'brown', 
    #marker_line_color = 'slategray', 
    #marker_line_width = 1.5, 
    opacity = 0.8
))
fig.add_trace(go.Bar(
    x = barplot_corruption_services['Entidad_Federativa'],
    y = barplot_corruption_services['Calificacion General'],
    name = 'Nivel de Satisfacción',
    marker_color = 'steelblue',
    #marker_line_color = 'slategray',
    #marker_line_width = 1.5, 
    opacity = 0.8
))
fig.update_layout(barmode = 'group', bargroupgap = 0.10, bargap = 0.20)
fig.update_layout(
legend = dict(orientation = "h", yanchor = "bottom", y = 1.02, xanchor = "right", x = 1
))
bargroupgap = 0.1

st.plotly_chart(fig) # satisfacción general con los servicios básicos

# Experiencias con Pagos, Trámites y Solicitudes de Servicios Públicos

st.write("# Experiencias con Pagos, Trámites y Solicitudes de Servicios Públicos")

services_provided = sec_7.loc[:, ['ENT', 'NOM_ENT', 'UPM', 'V_SEL', 'N_TRA', 'P7_4_1', 'P7_4_2', 'P7_4_3', 'P7_4_4', 'P7_4_5', 'P7_4_6', 'P7_4_7', 'P7_4_8', 'P7_4_9', 'P7_4_10', 'P7_4_11']] # columnas de interes correspondientes a los problemas principales que enfrenta la poblacion al momento de realizar un tramite de algun servicio 

services_provided = services_provided.set_index('NOM_ENT')

services_provided = services_provided.replace({'1': 'Si', '2': 'No', '3': 'No Aplica', '9': 'Indiferente', 'b': 'Sin Informacion'})

services_updated = services_provided.loc[((services_provided['N_TRA'] != '20') | (services_provided['N_TRA'] == '19')) & (services_provided['N_TRA'] != '21') & ((services_provided['N_TRA'] != '22a') & (services_provided['N_TRA'] != '22b') & (services_provided['N_TRA'] != '22c') & (services_provided['N_TRA'] != '22d'))].copy() # excluir trámites realizados por medios telefónicos 

services_updated = services_updated.iloc[:, 4:]

issues_related_services = ['Largas Filas', 'Falta de Claridad', 'Requisitos Excesivos', 'Ventanilla Incorrecta', 'Informacion Erronea', 'Fallas Telefonicas', 'Fallas Servicio en Linea', 'Lejania del Sitio', 'Costo Excesivo', 'Horario Restringido', 'Otro'] # instituciones publicas
services_updated.columns = issues_related_services

	# Problemas con Pagos, Trámites y Solicitudes de Servicios Públicos

categories = ['Si', 'No', 'No Aplica', 'Indiferente', 'Sin Informacion'] # posibles respuestas hacia la problematica en cuestion
contingency_table_issues_related_services = pd.DataFrame(columns = categories) # no. ocurrencias de cada nivel de percepcion en las distintas instituciones publicas

for column in services_updated.columns:
    frecuency = services_updated[column].value_counts().reindex(categories, fill_value = 0).to_dict() # ocurrencias de los distintos niveles de percepcion de la corrupcion (categorias) para las distintas instituciones
    contingency_table_issues_related_services = pd.concat([contingency_table_issues_related_services, pd.DataFrame(frecuency, index = [column])]) # anadir fila (institucion) con el no. de ocurrencias para las distintas categorias (columnas)

contingency_table_issues_related_services[categories] = contingency_table_issues_related_services[categories].astype(int) # conversion de object a int

contingency_table_issues_related_services = contingency_table_issues_related_services.iloc[:, :2]
contingency_table_issues_related_services['Total'] = contingency_table_issues_related_services.sum(axis = 1)

contingency_table_issues_related_services['% Si'] = (contingency_table_issues_related_services['Si'] / contingency_table_issues_related_services['Total']) * 100
contingency_table_issues_related_services['% No'] = (contingency_table_issues_related_services['No'] / contingency_table_issues_related_services['Total']) * 100
contingency_table_issues_related_services = contingency_table_issues_related_services.reset_index().sort_values(by = '% Si', ascending = False)

colors = ['lightslategray'] * 11
colors[0] = 'brown'
colors[1] = 'brown'
colors[2] = 'brown'

fig = go.Figure()
fig.add_trace(go.Bar(
    x = contingency_table_issues_related_services['index'],
    y = contingency_table_issues_related_services['% Si'],
    name = 'Frecuencia de Ocurrencia del Problema',
    marker_color = colors, 
    opacity = 0.8
))
fig.update_layout(barmode = 'group', bargroupgap = 0.10, bargap = 0.20)
bargroupgap = 0.1

st.plotly_chart(fig) # problemas con pagos, trámites y solicitudes de servicios públicos

st.write("## Problemas en el Pago, Trámite o Solicitud de Servicios de Salud")

services_provided = sec_7.loc[:, ['ENT', 'NOM_ENT', 'UPM', 'V_SEL', 'N_TRA', 'P7_4_1', 'P7_4_2', 'P7_4_3', 'P7_4_4', 'P7_4_5','P7_4_6', 'P7_4_7', 'P7_4_8', 'P7_4_9', 'P7_4_10', 'P7_4_11']] # columnas de interes correspondientes a los problemas principales que enfrenta la poblacion al momento de realizar un tramite de algun servicio 

services_provided = services_provided.set_index('NOM_ENT')

health_services_updated = services_provided.loc[(services_provided['N_TRA'] == '07') | (services_provided['N_TRA'] == '08')].copy()

health_services_updated = health_services_updated.iloc[:, 4:]

health_services_updated = health_services_updated.replace({'1': 'Si', '2': 'No', '3': 'No Aplica', '9': 'Indiferente', 'b': 'Sin Informacion'}) # reemplazar el valor numerico por la categoria correspondiente

issues_related_services = ['Largas Filas', 'Falta de Claridad', 'Requisitos Excesivos', 'Ventanilla Incorrecta', 'Informacion Erronea', 
              'Fallas Telefonicas', 'Fallas Servicio en Linea', 'Lejania del Sitio', 'Costo Excesivo', 'Horario Restringido', 'Otro'] # instituciones publicas
health_services_updated.columns = issues_related_services

categories = ['Si', 'No', 'No Aplica', 'Indiferente', 'Sin Informacion'] # posibles respuestas hacia la problematica en cuestion
contingency_table_health_services = pd.DataFrame(columns = categories) # no. ocurrencias de cada nivel de percepcion en las distintas instituciones publicas

for column in health_services_updated.columns:
    frecuency = health_services_updated[column].value_counts().reindex(categories, fill_value = 0).to_dict() # ocurrencias de los distintos niveles de percepcion de la corrupcion (categorias) para las distintas instituciones
    contingency_table_health_services = pd.concat([contingency_table_health_services, pd.DataFrame(frecuency, index = [column])]) # anadir fila (institucion) con el no. de ocurrencias para las distintas categorias (columnas)

contingency_table_health_services[categories] = contingency_table_health_services[categories].astype(int) # conversion de object a int

contingency_table_health_services_updated = contingency_table_health_services.iloc[:, :2]
contingency_table_health_services_updated['Total'] = contingency_table_health_services_updated.sum(axis = 1) # agregar columna con el total de entradas por fila

contingency_table_health_services_updated['% Si'] = (contingency_table_health_services_updated['Si'] / contingency_table_health_services_updated['Total']) * 100
contingency_table_health_services_updated['% No'] = (contingency_table_health_services_updated['No'] / contingency_table_health_services_updated['Total']) * 100
contingency_table_health_services_updated = contingency_table_health_services_updated.reset_index().sort_values(by = '% Si', ascending = False)

colors = ['lightslategray'] * 11
colors[0] = 'brown'

fig = go.Figure()
fig.add_trace(go.Bar(
    x = contingency_table_health_services_updated['index'],
    y = contingency_table_health_services_updated['% Si'],
    name = 'Frecuencia de Ocurrencia del Problema',
    marker_color = colors,  
    opacity = 0.8
))
fig.update_layout(barmode = 'group', bargroupgap = 0.10, bargap = 0.20)
bargroupgap=0.1

st.plotly_chart(fig) # problemas en el pago, trámite o solicitud de servicios de salud


