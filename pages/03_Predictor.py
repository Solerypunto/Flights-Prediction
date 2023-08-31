import streamlit as st
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
import random
import json

##### Configuracion de la página ###############################################################

st.set_page_config(page_title= 'Flight delay predictor',
                   page_icon= ':airplane_departure:',
                   initial_sidebar_state= 'expanded', layout= 'centered',)

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

### Data ##########################################

path = "Data/dataset_categorical_001.parquet"
df = pd.read_parquet(path)

path = "Data/df_ciudades.parquet"
df_ciudades = pd.read_parquet(path)

path = 'Data/DestWac-DestStateName.json'
with open(path, mode='r') as file:
    wac = json.load(file)
dfwac = pd.DataFrame(wac,columns=['key', 'values'])

##### Predictor ###############################################################

st.dataframe(df_ciudades)

st_lottie(requests.get("https://lottie.host/e94e8eb0-c1ed-41c3-b6da-b22a5104f594/4sofR9rtYA.json").json(), height=200, key="paperplane")

st.header('¿Cual es tu vuelo?')

# Imputs usuario
# Fecha
st.subheader('Fecha')
fechavuelo = st.date_input("¿Cuando es tu vuelo?", datetime.date(2023, 9, 10))
st.write('Tu vuelo sale el:', fechavuelo)
franjahora = st.select_slider(label='Franja horaria', options=['0900-0959', '1000-1059', '2200-2259', '1500-1559', '1100-1159',
                                                               '0800-0859', '1300-1359', '1600-1659', '1200-1259', '0700-0759',
                                                               '1400-1459', '0600-0659', '1700-1759', '2000-2059', '0001-0559',
                                                               '1900-1959', '1800-1859', '2100-2159', '2300-2359'])
st.divider()

# Origen
st.subheader('Origen')
estadoorigen = st.selectbox(label='Estado de origen', options=df_ciudades['State'].sort_values().unique())
Origin = df_ciudades[df_ciudades['State']==estadoorigen]['City'].sort_values().unique()
ciudadorigen = st.selectbox(label='Ciudad de destino', options=Origin)
aeropuerto = df_ciudades[df_ciudades['City']==ciudadorigen]['Origin'].tolist()
if len(aeropuerto) > 1:
    aeropuertoorigen = st.selectbox(label='Aeropuerto', options= aeropuerto)
else:
    aeropuertoorigen = aeropuerto[0]

st.divider()

# Destino
st.subheader('Destino')
destino = str(df_ciudades[df_ciudades['Origin'] == aeropuertoorigen]['Dest'].tolist()).split(' ')

ciudaddest = st.selectbox(label= 'Ciudad de destino', options = destino)



# estadodest = st.selectbox(label='Estado de destino', options=df['OriginStateName'])
# deswac = dfwac[dfwac['values']==estadodest]
# ciudaddest = st.selectbox(label='Ciudad de destino', options=Origin)
# aeropuerto = df.groupby([ciudaddest])[Origin]
# if len(aeropuerto) > 1:
#     aeropuertodest = st.selectbox(label='Aeropuerto', options= aeropuerto)
# else:
#     aeropuertodest=aeropuerto

# Aerolinea 
# 'Airline' > 'Operating_Airline'

# Opcionales

st.write(f'Tu vuelo: Origen {ciudadorigen}, {aeropuertoorigen} y destino {ciudaddest}, {aeropuertodest}, con fecha: {fechavuelo}')



Aerolinea = st.selectbox(label='Aerolinea', options = df['Airline'].unique())

dayofmonth = fechavuelo.day
month = fechavuelo.month
quarter = (fechavuelo.month - 1) // 3 + 1


# taxiout =

# horavuelo = st.date_input("¿Cuando es tu vuelo?", datetime.time())
# st.write('Tu vuelo sale a las:', horavuelo)

columnas = ["DepTimeBlkIndex", "Month", "DayofMonth", "Quarter", 
            "OriginIndex", "OriginAirportSeqID", "OriginStateNameIndex", "OriginCityMarketID", 
            "DestAirportSeqID", "DestStateNameIndex", "DestWac", 
            "Operating_AirlineIndex", "AirlineIndex", "Flight_Number_Operating_Airline", "DOT_ID_Operating_Airline", "DOT_ID_Marketing_Airline", 
            "Tail_NumberIndex", "TaxiOut", "WheelsOff", "WheelsOn", 
            "AirTime", "CRSElapsedTime", "ActualElapsedTime",]

