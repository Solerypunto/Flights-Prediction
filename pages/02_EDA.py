import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
from PIL import Image

##### Configuracion de la página ###############################################################

st.set_page_config(page_title = 'Flight delay predictor',
                   page_icon = ':airplane_departure:',
                   initial_sidebar_state = 'expanded', layout = 'centered',)

## CSS ###############################################################   
# tipografia   
streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Space Mono', sans-serif;

			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)

# imagenes redondeadas
st.markdown("""
    <style type="text/css">
    img {
    border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

## DATA ###############################################################
# dataset prueba
path = "Data/prueba_streamlit.csv"
df_all = pd.read_csv(path, sep=',', header= 0, )

# dataset ppal
path = "Data/dataset_categorical_001.parquet"
df_big = pd.read_parquet(path)
df_big_mapa = df_big[['Origin', 'OriginCityName', 'FlightDate', 'Dest','LATITUDE', 'LONGITUDE', 'DepDelayMinutes', 'DepDelay']]

# lat long
path = "Data/dic_lat_long.csv"
df_lat_long = pd.read_csv(path, sep=',', header= 0, )
path = 'Data/airports.csv'
airports_lat_long = pd.read_csv(path, sep=',', header= 0, )
airports_lat_long = airports_lat_long.drop(["AIRPORT", "CITY", "STATE", "COUNTRY"],axis=1)
df_lat_long = pd.concat([airports_lat_long,df_lat_long],axis=0)

# df Mapa 1
# Generamos un df con la columna 'Origin' para despues añadir las latitudes y longitudes de del origen
df_mapa1 = df_all[['Origin', 'OriginCityName']]
df_mapa1 = pd.merge(left=df_mapa1, right= df_lat_long, left_on= 'Origin',right_on='IATA',)
df_mapa1 = df_mapa1.drop('IATA',axis=1)
# Generamos un df con la columna 'Dest' para despues añadir las latitudes y longitudes de del destino. 
# Luego cambiamos nombres a las columnas de latitud y longitud
df_mapa11 = df_all[['Dest', 'DestCityName', 'DepDelayMinutes']]
df_mapa11 = pd.merge(left=df_mapa11, right= df_lat_long, left_on= 'Dest',right_on='IATA',)
df_mapa11 = df_mapa11.drop('IATA',axis=1)
df_mapa11 = df_mapa11.rename(columns= {'LATITUDE':'LATITUDEdest','LONGITUDE': 'LONGITUDEdest'},)
# Unimos ambos df (y borramos el df sobrante de la memoria)
df_mapa1 = pd.merge(left=df_mapa1, right= df_mapa11, left_index= True, right_index= True,)
del df_mapa11
df_mapa1 = df_mapa1.dropna()

# df Mapa 2 y 3
df_mapa = pd.DataFrame(df_big_mapa.groupby("Origin").agg(**{'max':('DepDelay','max'),'mean':('DepDelay','mean')}).reset_index())
df_mapa = df_mapa.round(1)
df_mapa = pd.merge(left=df_mapa, right= df_lat_long, left_on= 'Origin',right_on='IATA',)

#####################################################################################################################

st.header('EDA')

st_lottie(requests.get("https://lottie.host/e44dedf0-4f98-49e3-ab19-709275c763ae/zWrhP5d1F7.json").json(), height=300, key="airport-eda")

##
st.subheader('Dataset')
st.write('''El dataset contiene multiples documentos, se pueden agrupar en datos en crudo y datos ya ordenados, Optamos por quedarnos con los ordenados:
        5 documentos. \n
            Tamaño total del dataset: 29.193.782 x 61''')
st.write("")

# CARGA DE DATOS
st.markdown('#### Carga de datos')

imagen = Image.open("images/pyspark_data_load.png")
st.image(imagen, use_column_width = True)

st.write("Decidimos usar los archivos ```*.parquet``` porque eran más livianos que los ```*.csv```")
st.write("")

# STRINGINDEXER
st.markdown('#### StringIndexer()')

imagen = Image.open("images/string_indexer.png")
st.image(imagen, use_column_width = True)

st.write("Usamos ```StringIndexer()``` para transformar las columnas categóricas en numéricas.")
st.write("Tras varias pruebas con los datos en el siguiente notebook, nos percatamos de que el modelo de keras nos arrojaba métricas con valores null, y encontramos que el problema se solventaba añadiendo al indexer el parámetro ```.setHandleInvalid('skip')```. De esta forma omitimos los nulos que creaba, por algún motivo, el ```StringIndexer()```.")
st.write("")

# PREPROCESAMIENTO
st.markdown('#### Preprocesamiento de datos')

st.write("Usar ```.toPandas()``` con pyspark provocaba colapsos de memoria y terminaba rompiendo el kernel. Decidimos usarlo con límites y filtros para visualizar de alguna manera los datos en un dataframe.")
st.write("Para complementar hemos creado las columnas de ```latitud``` y ```longitud``` para hacer gráficas")

expander = st.expander(label = "Latitud y longitud", expanded = True)

with expander:
    st.dataframe(df_lat_long, use_container_width = True)

st.write("")

# OUTLIERS
st.markdown('#### Outliers')

st.write('''En función a las gráficas que generamos más abajo decidimos filtrar el dataframe para eliminar posibles outliers y quedarnos con los valores relevantes. \n
            Quitar outliers: df_all = df_all.filter(df_all["DepDelay"].between(-20, 200))''')

# FIN
st.write("")
st.write("")

st.write("Finalmente eliminamos columnas innecesarias y categóricas para dejar un DataFrame válido para usarlo en la predicción con el ```modelo de keras```.")

imagen = Image.open("images/df_final_100%_real_no_fake.png")
st.image(imagen, use_column_width = True)

st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

## Grafica 1
st.subheader('Bubble chart')
st.write( 'Para ver como se relaciona la media de retraso con el aeropuerto de origen. Cuanto mas grande sea el circulo mas alejado está de la media')

col_a, col_b = st.columns([1,2])

# with col_a: 
#     st.dataframe(df_all.groupby("Origin").agg(**{'max':('DepDelay','max'),'mean':('DepDelay','mean')}).reset_index(),use_container_width=True)

                                          
dfimpresionante = df_all.groupby("Origin").agg(**{'max':('DepDelay','max'),'mean':('DepDelay','mean')}).reset_index()

fig = px.scatter(dfimpresionante,
        x = 'Origin',
        y = "mean",
        size = dfimpresionante['max']+20,
        color = 'Origin')

st.plotly_chart(fig, theme='streamlit', use_container_width=True)


st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

## Correlacion
contenedor = st.container()

with contenedor:

    st.subheader('Matriz de correlación')
    
    corre = df_big[["DestStateNameIndex", "TaxiOut", "DepTimeBlkIndex", "WheelsOff", "DestWac", "AirTime", "CRSElapsedTime", "Operating_AirlineIndex",
                    "Month", "OriginStateNameIndex", "Flight_Number_Operating_Airline", "WheelsOn", "AirlineIndex", "DOT_ID_Marketing_Airline",
                    "ActualElapsedTime", "OriginAirportSeqID", "DOT_ID_Operating_Airline", "Quarter", "Tail_NumberIndex", "DestAirportSeqID", 
                    "OriginCityMarketID", "OriginIndex", "DayofMonth"]].corr()
    
    col_a, col_b, col_c = st.columns([0.5, 2, 0.5])

    # with col_a:
    #     st.write('Comprobamos la correlación de las columnas para acotar cuales usaremos en el modelo')
    #     st.dataframe(data = df_all, height = 600,)

    with col_b:
        fig, ax = plt.subplots(figsize=(24,24))
        sns.heatmap(corre, annot=True, ax=ax, cmap= 'plasma')
        st.pyplot(fig, use_container_width=True)

st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

## Histograma

st.subheader("Histograma")
st.write("Esta fue la gráfica en la que nos basamos para eliminar los outliers, la gran mayoría de los datos se encontraba entre ```-20 y 200```. Todo esto ayudó a mejorar las métricas posteriores.")
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12))

# Primera columna
ax[0, 0].hist(df_big[df_big["Operating_AirlineIndex"] == 0.0]["DepDelay"], bins = 100, color = "cyan")   

ax[1, 0].hist(df_big[df_big["Operating_AirlineIndex"] == 1.0]["DepDelay"], bins = 100, color = "black")

# Segunda columna
ax[0, 1].hist(df_big[df_big["Operating_AirlineIndex"] == 21.0]["DepDelay"], bins = 100, color = "green")   

ax[1, 1].hist(df_big[df_big["Operating_AirlineIndex"] == 13.0]["DepDelay"], bins = 100, color = "red")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

## Pie

st.subheader("Pie chart")
st.write("Aquí visualizamos como afecta el retraso según la hora de salida de los vuelos. Una vez ```avanza``` el día, los retrasos se acumulan. Las horas ```posteriores a la mañana``` y ```previas a la noche``` no son aconsejables porque son más propensas a retrasos.")

fig = px.pie(data_frame = df_big,
             names = "DepTimeBlk",
             values = "DepDelay")

st.plotly_chart(figure_or_data = fig, use_container_width = True)

st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

## Mapa 3D

st.subheader('Mapa Retrasos')
st.write('En este mapa podemos ver la media de los retrasos en los aeropuertos de EUA. \n A mayor retraso, mas altura en la columna')

# https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart 

Y_RGB = [255, 255, 0, 80]

st.pydeck_chart(pdk.Deck( map_style=None, 
                            initial_view_state=pdk.ViewState(latitude=38,
                                                            longitude= -98.579437, 
                                                            zoom=2.6,
                                                            pitch=50,),
                            layers=[pdk.Layer("ColumnLayer",
                                            data=df_mapa,
                                            get_position='[LONGITUDE,LATITUDE]',
                                            get_elevation='mean',
                                            radius=30000,
                                            elevation_scale=7500,
                                            elevation_range=[0, 1000], 
                                            get_fill_color= Y_RGB,
                                            auto_highlight=True,
                                            pickable=True,
                                            extruded=True,
                                            ),],tooltip= {
"html": "Aeropuerto <b>{Origin}</b>. <br>  <b>{mean}</b> min de retrasio medio <br> <b>{max}</b> min. de retraso máximo. ",
"style": {"background": "black", "color": "white", "font-family": '"Space Mono", Arial', "z-index": "8000",  'border-radius': '5'}},))

# mapa de calor
st.pydeck_chart(pdk.Deck( map_style=None, 
                            initial_view_state=pdk.ViewState(latitude=38,
                                                            longitude= -98.579437, 
                                                            zoom=2.6,
                                                            pitch=0,),
                            layers=[pdk.Layer("HeatmapLayer",
                                            data=df_mapa,
                                            get_position='[LONGITUDE,LATITUDE]',
                                            aggregation='mean',
                                            get_weight='max',
                                            pickable=True,
                                            opacity=0.8,
                                            ),],
                                            tooltip= {
"html": "Aeropuerto <b>{Origin}</b>. <br>  <b>{mean}</b> min de retrasio medio <br> <b>{max}</b> min. de retraso máximo. ",
"style": {"background": "black", "color": "white", "font-family": '"Space Mono", Arial', "z-index": "8000",  'border-radius': '5'}},
                                            ))

st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)