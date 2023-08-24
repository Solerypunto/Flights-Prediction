import streamlit as st
import webbrowser
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

##### Configuracion de la página ###############################################################

st.set_page_config(page_title= 'Flight delay predictor',
                   page_icon= ':airplane_departure:',
                   initial_sidebar_state= 'collapsed', layout= 'wide',)



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
# imagenes redondas
# st.markdown("""
#             <style type="text/css">
#             img {
#             border-radius: 10000px;
#             }
#             </style>
#             """, unsafe_allow_html=True)



## DATA ###############################################################
# dataset ppal
path = "Data/prueba_streamlit.csv"
df_all = pd.read_csv(path, sep=',', header= 0, )

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
df_mapa = pd.DataFrame(df_all.groupby("Origin").agg(**{'max':('DepDelay','max'),'mean':('DepDelay','mean')}).reset_index())
df_mapa = df_mapa.round(1)
df_mapa = pd.merge(left=df_mapa, right= df_lat_long, left_on= 'Origin',right_on='IATA',)

##### Cuerpo de la página ###############################################################
def main():

    st.title('FLIGHT DELAY PREDICTOR')

    home, eda, predictor, about = st.tabs(['Home', 'EDA', 'Predictor', 'About'])



##### Home ###############################################################

    with home:
        st.header('Esto es un :orange[header] de prueba')
        st.subheader('Esto es un subheader pa probá a ve como queda esto')
        st. write('''Aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, ''')
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

        # Mapa vuelos
        
        ORIGENES = list(df_mapa1['OriginCityName'].unique())
        ORIGEN = st.multiselect(label = "Ciudad de origen",
                                  options = ORIGENES, 
                                  default = 'New York, NY')

        dfmapapersonalizado = df_mapa1[df_mapa1['OriginCityName'].isin(ORIGEN)]
        dfmapapersonalizado['DepDelayMinutes'] += 10

        GREEN_RGB = [255, 255, 0, 40]
        RED_RGB = [240, 100, 0, 40]

        st.pydeck_chart(pdk.Deck( map_style=None, 
                                 initial_view_state=pdk.ViewState(latitude=38,
                                                                  longitude= -98.579437, 
                                                                  zoom=2.6,
                                                                  pitch=0,),
                                 layers=[pdk.Layer("ArcLayer",
                                                    data= dfmapapersonalizado,
                                                    get_width="DepDelayMinutes /100",
                                                    get_source_position=["LONGITUDE", "LATITUDE"],
                                                    get_target_position=["LONGITUDEdest", "LATITUDEdest"],
                                                    get_tilt=1,
                                                    get_source_color=GREEN_RGB,
                                                    get_target_color=RED_RGB,
                                                    pickable=True,
                                                    auto_highlight=True,
                                                    ),],
                                                    tooltip= {
        "html": "Origen <b>{OriginCityName}</b>, Destino <b>{DestCityName}</b>. <br>  <b>{DepDelayMinutes}</b> min de retrasio.",
        "style": {"background": "black", "color": "white", "font-family": '"Space Mono", Arial', "z-index": "8000",  'border-radius': '5'}},
                                                    ))
        del dfmapapersonalizado


##### EDA ###############################################################

    with eda:
        ##
        st.subheader('Dataset')
        st.write('''El dataset contiene multiples documentos, se pueden agrupar en datos en crudo y datos ya ordenados, Optamos por quedarnos con los ordenados:
                6 documentos. Optamos por el formato *.parquet porque es mas ligero que *.csv \n
                 Tamaño total del dataset: 29.193.782 x 61''')
        st.write('Para complementar hemos creado las columnas de Latitud y Longitud para hacer gráficas')

        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)
        
        ## Grafica 1
        st.subheader('Gráfica 1')
        st.write( 'Para ver como se relaciona la media de retraso con el aeropuerto de origen. Cuanto mas grande sea el circulo mas alejado está de la media')

        col_a, col_b = st.columns([1,2])

        with col_a: 
            st.dataframe(df_all.groupby("Origin").agg(**{'max':('DepDelay','max'),'mean':('DepDelay','mean')}).reset_index(),use_container_width=True)

        with col_b:                                           
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

            corre = df_all.drop(['FlightDate', 'Airline', 'Origin', 'Dest', 'Cancelled', 'Diverted', 'Marketing_Airline_Network', 
                                    'Operated_or_Branded_Code_Share_Partners', 'IATA_Code_Marketing_Airline', 'Operating_Airline', 
                                    'IATA_Code_Operating_Airline', 'Tail_Number', 'OriginCityName', 'OriginState', 'OriginStateName',
                                    'DestCityName', 'DestState', 'DestStateName', 'ArrTimeBlk', 'DepTimeBlk', 'Year', 'Quarter', 'Month'], axis= 1).sample(frac=0.01).corr()
            
            col_a, col_b = st.columns(2)

            with col_a:
                st.write('Comprobamos la correlación de las columnas para acotar cuales usaremos en el modelo')
                st.dataframe(data = df_all, height = 600,)

            with col_b:
                fig, ax = plt.subplots(figsize=(24,24))
                sns.heatmap(corre, annot=True, ax=ax, cmap= 'plasma')
                st.pyplot(fig, use_container_width=True)


        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)
        
        ## Mapa 3D

        st.subheader('Mapa Retrasos')
        st.write('En este mapa podemos ver la media de los retrasos en los aeropuertos de EUA. \n A mayor retraso, mas altura en la columna')

        # https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart 

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
                                                    get_fill_color= [255, 255, 0 , 75],
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



##### About ###############################################################

    with about:
        st.header('Team')
        col_1, col_2, col_3 = st.columns(3)

        with col_1:
            st.image('https://media.licdn.com/dms/image/D4D03AQEZNDYmDw6xbQ/profile-displayphoto-shrink_400_400/0/1687197904792?e=1697673600&v=beta&t=punW-e_QYI9KavGDk8XWqvzckRiuNT9yzGn3d1BDfvY',
                    width= 200,)
            st.subheader('Germán Fernandez')
            # if col_1.button(label = 'Linkedin Germán', use_container_width= True, ):
            #     webbrowser.open_new_tab('https://www.linkedin.com/in/german-fernandez-corrales/')
            link = '[Linkedin](https://www.linkedin.com/in/german-fernandez-corrales/)'
            st.markdown(link, unsafe_allow_html=True)

        with col_2:
            st.image('https://media.licdn.com/dms/image/D4D03AQHaDZ_9s0lxmw/profile-displayphoto-shrink_400_400/0/1686648949588?e=1697673600&v=beta&t=ICAUdXK_vGrLMuzRQVqEmNW5rmQai1INiEzbSkI9LLY',
                    width= 200,)
            st.subheader('Miguel Nieto')
            # if col_2.button(label = 'Linkedin Miguel', use_container_width= True, ):
            #     webbrowser.open_new_tab('https://www.linkedin.com/in/miguel-nieto-p/')
            link = '[Linkedin](https://www.linkedin.com/in/miguel-nieto-p/)'
            st.markdown(link, unsafe_allow_html=True)
        
        with col_3:
            st.image('https://media.licdn.com/dms/image/D4D03AQGJSyQ4v4QVFw/profile-displayphoto-shrink_800_800/0/1687511681863?e=2147483647&v=beta&t=kJ_dBjX9dpThCF6CfqgSGo8R-9j8hNSJTbXQDewXHYU',
                    width= 200,)
            st.subheader('Sergio Soler')
            # if col_3.button(label = 'Linkedin Sergio', use_container_width= True, ):
            #     webbrowser.open_new_tab('https://www.linkedin.com/in/sergiosolergarcia/')
            link = '[Linkedin](https://www.linkedin.com/in/sergiosolergarcia/)'
            st.markdown(link, unsafe_allow_html=True)

    pass

if __name__ == '__main__':
    main()

