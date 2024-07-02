import pandas as pd
import numpy as np
import os
import geopandas as gpd
import matplotlib.pyplot as plt

data_dir = '/home/victor/cursos/ciencia_de_datos_general/'

# lee la cartografía de entidades y municipios de México
fileindex = os.path.join(data_dir, 'data/indice_marginacion/cartografia_nacional_inegi2022/national/national_estatal.shp')
base = gpd.read_file(fileindex)
fileindex = os.path.join(data_dir, 'data/indice_marginacion/cartografia_nacional_inegi2022/mg2021_integrado_tarea/conjunto_de_datos/00mun.shp')
layer = gpd.read_file(fileindex, index_col='CVEGEO')
layer = layer.to_crs("EPSG:4326") # corrige el sistema de coordenadas de referencia para los municipios

# datos del índice de marginación
fileindex = os.path.join(data_dir, 'data/indice_marginacion/IMM_2020.xls')
# especifica el tipo de variable string para los códigos de entidades y municipios
lst_str_cols = ['CVE_ENT', 'NOM_ENT', 'CVE_MUN', 'NOM_MUN']
dict_dtypes = {x : 'str'  for x in lst_str_cols}
marg_municipal = pd.read_excel(fileindex,sheet_name='IMM_2020', dtype=dict_dtypes)
marg_municipal = marg_municipal.set_index('CVE_MUN')
# une la información cartográfica y del IM
layer_marg = layer.merge(marg_municipal, left_on='CVEGEO',right_on='CVE_MUN')
# especifica la variable GM_2020 como categórica (el IM en 5 niveles)
im_cat = pd.CategoricalIndex(layer_marg['GM_2020'], categories=['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto'],)

# grafica el indicador a nivel municipal
f, ax = plt.subplots(1, figsize=(15, 15))
ax.set_aspect('equal')
# mapa de las entidades de México
base.plot(color='white', edgecolor='black', ax=ax) 
# mapa de los municipios con su IM
layer_marg.plot(column='GM_2020', categorical=True, legend=True, linewidth=0, ax=ax, cmap="Reds", 
           categories = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto']) 
# Remove axis
ax.set_axis_off()
plt.show()

