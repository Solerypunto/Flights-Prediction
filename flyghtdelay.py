import streamlit as st
import webbrowser
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

##### Configuracion de la página ###############################################################

st.set_page_config(page_title= 'Flight delay predictor',
                   page_icon= ':airplane_departure:',
                   initial_sidebar_state= 'expanded', layout= 'wide',)



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


## DATA ###############################################################
# dataset prueba
path = "Data/prueba_streamlit.csv"
df_all = pd.read_csv(path, sep=',', header= 0, )

# dataset ppal
path = "Data/dataset_categorical_001.parquet"
df_big = pd.read_parquet(path)

# # lat long
# path = "Data/dic_lat_long.csv"
# df_lat_long = pd.read_csv(path, sep=',', header= 0, )
# path = 'Data/airports.csv'
# airports_lat_long = pd.read_csv(path, sep=',', header= 0, )
# airports_lat_long = airports_lat_long.drop(["AIRPORT", "CITY", "STATE", "COUNTRY"],axis=1)
# df_lat_long = pd.concat([airports_lat_long,df_lat_long],axis=0)

# # df Mapa 1
# # Generamos un df con la columna 'Origin' para despues añadir las latitudes y longitudes de del origen
# df_mapa1 = df_all[['Origin', 'OriginCityName']]
# df_mapa1 = pd.merge(left=df_mapa1, right= df_lat_long, left_on= 'Origin',right_on='IATA',)
# df_mapa1 = df_mapa1.drop('IATA',axis=1)
# # Generamos un df con la columna 'Dest' para despues añadir las latitudes y longitudes de del destino. 
# # Luego cambiamos nombres a las columnas de latitud y longitud
# df_mapa11 = df_all[['Dest', 'DestCityName', 'DepDelayMinutes']]
# df_mapa11 = pd.merge(left=df_mapa11, right= df_lat_long, left_on= 'Dest',right_on='IATA',)
# df_mapa11 = df_mapa11.drop('IATA',axis=1)
# df_mapa11 = df_mapa11.rename(columns= {'LATITUDE':'LATITUDEdest','LONGITUDE': 'LONGITUDEdest'},)
# # Unimos ambos df (y borramos el df sobrante de la memoria)
# df_mapa1 = pd.merge(left=df_mapa1, right= df_mapa11, left_index= True, right_index= True,)
# del df_mapa11
# df_mapa1 = df_mapa1.dropna()

# # df Mapa 2 y 3
# df_mapa = pd.DataFrame(df_all.groupby("Origin").agg(**{'max':('DepDelay','max'),'mean':('DepDelay','mean')}).reset_index())
# df_mapa = df_mapa.round(1)
# df_mapa = pd.merge(left=df_mapa, right= df_lat_long, left_on= 'Origin',right_on='IATA',)

##### Cuerpo de la página ###############################################################
def main():

    # st.title('FLIGHT DELAY PREDICTOR')

    with st.sidebar:

        selected = option_menu(None, ['HOME', 'EDA', 'PREDICTOR', 'ABOUT'], 
            icons=['house', 'bi bi-graph-up','bi bi-airplane','list'], 
            menu_icon="cast", default_index=0, 
            styles={
                "nav-link-selected": {"color": "#000000", 'font-weight': '900'},
                "nav-link": {"--hover-color": "#1E1E1C", 'font-weight': '900'},
                
                },)

##### Home ###############################################################

    if selected == 'HOME':
 
        st_lottie(requests.get("https://lottie.host/af7dcf94-5405-4391-90c5-e4cb9f801dbb/ZcNtRXN2y4.json").json(), height=300, key="airport")

        st.header('Predicción en el retraso de los vuelos')
        st.subheader('A partir de dataset de vuelos intra-estadounidenses')
        st. write('''Aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, ''')
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

        # Mapa vuelos
        
        ORIGENES = list(df_big['OriginCityName'].head(200).unique())
        ORIGEN = st.multiselect(label = "Ciudad de origen",
                                options = ORIGENES, 
                                default = 'New York, NY'
                                )

        dfmapapersonalizado = df_big[df_big[['OriginCityName']].isin(ORIGEN)]
        # dfmapapersonalizado['DepDelayMinutes'] += 10

        Y_RGB = [255, 255, 0, 40]
        G_RGB = [56, 191, 140, 40]

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
                                                    get_source_color=Y_RGB,
                                                    get_target_color=G_RGB,
                                                    pickable=True,
                                                    auto_highlight=True,
                                                    ),],
                                                    tooltip= {
        "html": "Origen <b>{OriginCityName}</b>, Destino <b>{DestCityName}</b>. <br>  <b>{DepDelayMinutes}</b> min de retrasio.",
        "style": {"background": "black", "color": "white", "font-family": '"Space Mono", Arial', "z-index": "8000",  'border-radius': '5'}},
                                                    ))
        del dfmapapersonalizado


##### Predictor ###############################################################

    if selected == 'PREDICTOR':

        st_lottie(requests.get("https://lottie.host/e94e8eb0-c1ed-41c3-b6da-b22a5104f594/4sofR9rtYA.json").json(), height=200, key="paperplane")

        st.header('¿Cual es tu vuelo?')

        fechavuelo = st.date_input("¿Cuando es tu vuelo?", datetime.date(2023, 9, 10))
        st.write('Tu vuelo sale el:', fechavuelo)

        horavuelo = st.date_input("¿Cuando es tu vuelo?", datetime.time())
        st.write('Tu vuelo sale a las:', horavuelo)

    pass

if __name__ == '__main__':
    main()

