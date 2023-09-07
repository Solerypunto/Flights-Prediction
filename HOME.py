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
                   initial_sidebar_state= 'expanded', layout= 'centered')


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

    st.header('Despega Con Calma: Predice Tu Vuelo 🛫')
    st.subheader('Predicción en el retraso de los vuelos a partir de dataset de vuelos intra-estadounidenses')
    st.write('¿Alguna vez has soñado con la posibilidad de despedirte de ese madrugón innecesario para atrapar un vuelo temprano? ¿O quizás deseas tener tiempo de sobra para hacer las maletas sin prisas? Buenas noticias, ¡tu búsqueda ha terminado! Nos complace presentarte nuestro proyecto: "Despega Con Calma: Predice Tu Vuelo".')
    st.write('Imagina un mundo donde tus próximas aventuras comienzan sin estrés ni sorpresas desagradables en el aeropuerto. Eso es exactamente lo que hemos estado trabajando arduamente para lograr. Nos embarcamos en el mundo de los datos de vuelos en Estados Unidos, recopilando información desde 2018 hasta mediados de 2022. Sí, hablamos de 29 millones de filas y 61 columnas de datos en un emocionante desafío para mantener nuestros vuelos en horario.')
    st.write('A los mandos, los carismáticos Sergio Soler, Germán Fernández y Miguel Nieto, y Daniel Tümmler en la torre de control con sus tutorías. Nuestro equipo se aventuró en el bootcamp de Data Science de Hack A Boss y este proyecto es el gran broche de oro de nuestro viaje. ¿Nuestra misión? Utilizar Pyspark, Pandas, Scikit-Learn, TensorFlow, Keras y muchas otras herramientas, para crear un modelo de predicción de retraso de vuelos que sea tan confiable como el piloto automático de tu avión favorito.')
    st.write('¿Qué te espera en nuestra web de Streamlit? Bueno, tenemos algunas pestañas curiosas para ti. En "EDA" (Exploratory Data Analysis), te sumergirás en visualizaciones que revelarán los secretos ocultos en los datos, ayudándonos a entrenar nuestro modelo de manera eficiente. En "Predictor", encontrarás una experiencia interactiva donde podrás ingresar tus detalles de vuelo y descubrir si te espera un día relajado o una carrera contra el tiempo.')
    st.write('Luego, en "Modelo", te revelaremos los misterios detrás de la magia. Desglosaremos los pasos que seguimos para entrenar a nuestro modelo, cómo identificamos las columnas más cruciales y las métricas que usamos para medir su desempeño. Y para terminar, en "About", te conectaremos con nosotros a través de LinkedIn y GitHub, para que puedas seguir nuestras aventuras en el mundo de los datos.')
    st.write('¡Bienvenido a bordo de "Despega Con Calma: Predice Tu Vuelo"! 🌟✈️')
    
    st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)
    
    kaggle = '[Flight Status Prediction | Kaggle](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022)'
    st.markdown(kaggle, unsafe_allow_html=True)
    
    # if st.button(':bar_chart: Dataset (Kaggle)'):
    #     webbrowser.open_new_tab('https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022?select=Combined_Flights_2022.csv')
    
if __name__ == '__main__':
    main()

