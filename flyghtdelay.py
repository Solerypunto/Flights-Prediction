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

path = "Data/dataset_1Mx8_streamlit.parquet"
df_mapa_vuelos = pd.read_parquet(path)

path = "Data/dataset_groupby_origin_streamlit.parquet"
df_mapas = pd.read_parquet(path)

##### Cuerpo de la página ###############################################################
def main():

    # st.title('FLIGHT DELAY PREDICTOR')
    st.sidebar.title('Predictor de retraso de vuelo')

    # with st.sidebar:

    #     selected = option_menu(None, ['HOME', 'EDA', 'PREDICTOR', 'ABOUT'], 
    #         icons=['house', 'bi bi-graph-up','bi bi-airplane','list'], 
    #         menu_icon="cast", default_index=0, 
    #         styles={
    #             "nav-link-selected": {"color": "#000000", 'font-weight': '900'},
    #             "nav-link": {"--hover-color": "#1E1E1C", 'font-weight': '900'},
                
    #             },)

##### Home ###############################################################


 
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
    
    st.dataframe(df_mapas, use_container_width= True)


    # ORIGENES = list(df_mapa_vuelos['OriginCityName'].head(200).unique())
    # ORIGEN = st.multiselect(label = "Ciudad de origen",
    #                         options = ORIGENES, 
    #                         # default = 'New York, NY'
    #                         )

    # dfmapapersonalizado = df_mapa_vuelos[df_mapa_vuelos[['OriginCityName']].isin(ORIGEN)]
    # dfmapapersonalizado['DepDelayMinutes'] += 10

    # Y_RGB = [255, 255, 0, 40]
    # G_RGB = [56, 191, 140, 40]

    # st.pydeck_chart(pdk.Deck( map_style=None, 
    #                          initial_view_state=pdk.ViewState(latitude=38,
    #                                                           longitude= -98.579437, 
    #                                                           zoom=2.6,
    #                                                           pitch=0,),
    #                          layers=[pdk.Layer("ArcLayer",
    #                                             data= dfmapapersonalizado,
    #                                             get_width="DepDelayMinutes /100",
    #                                             get_source_position=["LONGITUDE", "LATITUDE"],
    #                                             get_target_position=["LONGITUDEdest", "LATITUDEdest"],
    #                                             get_tilt=1,
    #                                             get_source_color=Y_RGB,
    #                                             get_target_color=G_RGB,
    #                                             pickable=True,
    #                                             auto_highlight=True,
    #                                             ),],
    #                                             tooltip= {
    # "html": "Origen <b>{OriginCityName}</b>, Destino <b>{DestCityName}</b>. <br>  <b>{DepDelayMinutes}</b> min de retrasio.",
    # "style": {"background": "black", "color": "white", "font-family": '"Space Mono", Arial', "z-index": "8000",  'border-radius': '5'}},
    #                                             ))
    # del dfmapapersonalizado

    pass

if __name__ == '__main__':
    main()

