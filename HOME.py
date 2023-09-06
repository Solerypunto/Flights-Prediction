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
import webbrowser

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


##### Cuerpo de la página ###############################################################
def main():

    # st.sidebar.title('Predictor de retraso de vuelo')


#####################################################################################################################
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

    if st.button(':bar_chart: Dataset (Kaggle)'):
        webbrowser.open_new_tab('https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022?select=Combined_Flights_2022.csv')
    
if __name__ == '__main__':
    main()

