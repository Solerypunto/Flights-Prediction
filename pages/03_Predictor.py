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
import json
import math
import pickle as pkl
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import random

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

st_lottie(requests.get("https://lottie.host/e94e8eb0-c1ed-41c3-b6da-b22a5104f594/4sofR9rtYA.json").json(), height=200, key="paperplane")

st.header('¿Cual es tu vuelo?')

### Imputs usuario
## Fecha
st.subheader('Fecha')
fechavuelo = st.date_input("¿Cuando es tu vuelo?", datetime.date(2023, 9, 10))

franjahora = st.select_slider(label='Franja horaria', options=sorted(['0900-0959', '1000-1059', '2200-2259', '1500-1559', '1100-1159',
                                                               '0800-0859', '1300-1359', '1600-1659', '1200-1259', '0700-0759',
                                                               '1400-1459', '0600-0659', '1700-1759', '2000-2059', '0001-0559',
                                                               '1900-1959', '1800-1859', '2100-2159', '2300-2359']))

st.divider()


## Origen
st.subheader('Origen')

estadoorigen = st.selectbox(label='Estado de origen', options=df_ciudades['State'].sort_values().unique())
Origin = df_ciudades[df_ciudades['State']==estadoorigen]['City'].sort_values().unique()
ciudadorigen = st.selectbox(label='Ciudad de origen', options=Origin)
aeropuerto = df_ciudades[df_ciudades['City']==ciudadorigen]['Origin'].tolist()
if len(aeropuerto) > 1:
    aeropuertoorigen = st.selectbox(label='Aeropuerto', options= aeropuerto)
else:
    aeropuertoorigen = aeropuerto[0]

st.divider()


## Destino
st.subheader('Destino')
aeropuertodestino = df_ciudades[df_ciudades['Origin'] == aeropuertoorigen]['Dest'].tolist()[0].tolist()

ciudaddestino = list()
for i in aeropuertodestino:
    ciudaddestino.append(df_ciudades[df_ciudades['Origin'] == i]['City'].values[0])

ciudaddest = st.selectbox(label= 'Ciudad de destino', options = sorted(set(ciudaddestino)))
aeropuerto_destino = df_ciudades[(df_ciudades['City']== ciudaddest) & 
                                 (df_ciudades[df_ciudades['City']==ciudaddest]['Origin'].unique().any() in aeropuertodestino)]['Origin'].tolist()
if len(aeropuerto_destino) > 1: 
    a_dest = st.selectbox(label='Aeropuerto de destino', options=aeropuerto_destino)
else:
    a_dest = aeropuerto_destino[0]
st.divider()

## Vuelo
st.subheader('Vuelo')

airline = sorted(df[(df['Origin']==aeropuertoorigen) & (df['Dest']==a_dest)]['Airline'].unique())
Aerolinea = st.selectbox(label='Aerolinea', options = airline)

Numerovuelo = sorted(df[(df["Origin"] == aeropuertoorigen) & (df["Dest"] == a_dest) & (df["Airline"] == Aerolinea)]["Flight_Number_Operating_Airline"].unique())
Numero_vuelo = st.selectbox(label='Numero de vuelo', options=Numerovuelo)

st.divider()

## Opcionales 
st.subheader('Valores opcionales')

opcionales = st.expander(label = 'Estos valores son para jugar :)')

matricula = df[df['Airline']==Aerolinea]['Tail_Number'].unique().tolist()
taxiout = sorted(df[df['Airline']==Aerolinea]['TaxiOut'].unique().tolist())

with opcionales:
    taxi_out = st.select_slider(label= 'Taxi Out: Tiempo del avion en pista antes de despegar (en minutos)', options= taxiout)
    tail_number = st.selectbox(label= 'Elige tu avion', options= matricula, index=int(len(matricula)/2))

st.divider()

# distancia recorrida
lat1 = float(df[df['Origin']== aeropuertoorigen]['LATITUDE'].unique()[0])
lon1 = float(df[df['Origin']== aeropuertoorigen]['LONGITUDE'].unique()[0])
lat2 = float((df[df['Origin']== a_dest]['LATITUDE'].unique()[0]))
lon2 = float(df[df['Origin']== a_dest]['LONGITUDE'].unique()[0])
# Calcular la diferencia entre las latitudes y longitudes.
d_lat = lat2 - lat1
d_lon = lon2 - lon1
# Calcular la distancia entre los dos puntos.
a = math.sin(d_lat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon/2)**2
distance = 2 * 6371 * math.asin(math.sqrt(a))
# Tiempo en el aire
velociddecrucero = 840
tiempoaire = distance/velociddecrucero
tiempotaxi = float(taxi_out)+float(df[df['Airline']==Aerolinea]['TaxiOut'].mode())
tiempovuelo = tiempoaire+(taxi_out+float(df[df['Airline']==Aerolinea]['TaxiOut'].mode()))/60
tiempodevueloreal = tiempovuelo ## CALCULAR MARGEN


## Resumen
st.subheader('''Tu vuelo \n 
            Resumen de tu vuelo:\n
            Origen: {} ({}) \n 
            Destino: {} ({}) \n 
            Fecha: {}\n
            Distancia a recorrer: {}km\n
            Duracion de viaje: {}h\n
            Tiempo en el aire: {}h'''.format(ciudadorigen, aeropuertoorigen, ciudaddest,
                                             a_dest, fechavuelo, round(distance, 2), round(tiempovuelo, 2), 
                                             round(tiempoaire, 2)))

# Generamos Variables

dayofmonth = fechavuelo.day
month = fechavuelo.month
quarter = (fechavuelo.month - 1) // 3 + 1
estadodestino = df[df['Dest']==a_dest]['DestStateName'].unique()[0]
estadodestinoindex = df[df['Dest']==a_dest]['DestStateIndex'].unique()[0]
des_wac = df[df['DestStateIndex']==estadodestinoindex]['DestWac'].unique()[0]
operating_airline = df[df['Airline']==Aerolinea]["Operating_AirlineIndex"].unique()[0]
airline_index = df[df['Airline']==Aerolinea]["AirlineIndex"].unique()[0]
originstateindex = df[df['Origin']==aeropuertoorigen]['OriginStateNameIndex'].unique()[0]
dot_id = df[df['Airline']==Aerolinea]["DOT_ID_Marketing_Airline"].unique()[0]
OriginSeq = df[df['Origin']==aeropuertoorigen]["OriginAirportSeqID"].unique()[0]
dot_id_operating =  df[df['Airline']==Aerolinea]["DOT_ID_Operating_Airline"].unique()[0]
DestSeq = df[df['Dest']==a_dest]["DestAirportSeqID"].unique()[0]
OriginCityMarket = df[df['OriginCityName']==ciudadorigen]["OriginCityMarketID"].unique()[0]
Origin_ID = df[df['Origin']==aeropuertoorigen]['OriginIndex'].unique()[0]
franjahoraindex = df[df['DepTimeBlk']==franjahora]['DepTimeBlkIndex'].unique()[0]
tailid = df[df['Tail_Number']==tail_number]['Tail_NumberIndex'].unique()[0]

####### LA TRAMPA ###################################################################
uso_1 = 1
uso_2 = 2
uso_3 = 3
uso_4 = 4
uso_5 = 5
uso_6 = 6
uso_7 = 21

usos_horarios= {'Hawaii': uso_1, 'Alasca': uso_2, 'Washington': uso_3, 'Oregon': uso_3, 'Nevada': uso_3, 'California': uso_3,
                 'Montana': uso_4, 'Idaho': uso_4, 'Utah': uso_4, 'Wyoming': uso_4,'Colorado': uso_4, 'Arizona': uso_4, 'New Mexico': uso_4,
                 'North Dakota': uso_5, 'South Dakota': uso_5, 'Nebraska': uso_5, 'Kansas': uso_5, 'Oklahoma': uso_5, 'Texas': uso_5, 
                 'Minnesota': uso_5, 'Iowa': uso_5, 'Missouri': uso_5, 'Arkansas': uso_5, 'Louisiana': uso_5,'Wisconsin': uso_5,
                 'Illinois': uso_5, 'Tenessee': uso_5, 'Mississippi': uso_5, 'Alabama': uso_5, 'Michigan': uso_6, 'Indiana': uso_6, 'Kentucky': uso_6, 
                 'Georgia': uso_6, 'Florida': uso_6, 'Ohio': uso_6, 'West Virginia': uso_6, 'Virginia': uso_6, 'North Carolina': uso_6, 
                 'South Carolina': uso_6, 'Pennsylvania': uso_6, 'Delaware': uso_6, 'New York': uso_6, 'Vermont': uso_6, 'Maine': uso_6, 
                 'Maryland': uso_6, 'New Jersey': uso_6, 'New Hampshire': uso_6, 'Massachusetts': uso_6, 'Rhode Island': uso_6, 'Connecticut': uso_6, 
                 'Puerto Rico': uso_6, 'U.S. Virgin Islands': uso_6, 'U.S. Pacific Trust Territories and Possessions': uso_7}

blktime_1 = datetime.datetime.strptime(str(int(franjahora.split('-')[1]) - random.randint(0, 59)), "%H%M")
blktime_2 = (datetime.datetime.strptime(str(franjahora.split('-')[1]), "%H%M") + datetime.timedelta(minutes=random.randint(0, 59)))
wheels_off = random.choice([blktime_1, blktime_2])
wheels_on = wheels_off + datetime.timedelta(hours = (int(usos_horarios.get(estadoorigen)) - int(usos_horarios.get(estadodestino))), minutes = int(tiempoaire*60))
wheels_off = int(datetime.time.strftime(wheels_off.time(),'%H%M'))
wheels_on = int(datetime.time.strftime(wheels_on.time(),'%H%M'))

### PREDICCION ######################################################################
## generamos el vectotr
X = [[estadodestinoindex, taxi_out, franjahoraindex, wheels_off, float(des_wac), tiempoaire*60, tiempovuelo*60, operating_airline, month, originstateindex, 
                  float(Numero_vuelo),float(wheels_on), airline_index, float(dot_id), tiempodevueloreal*60, float(OriginSeq), float(dot_id_operating), 
                  quarter, tailid, float(DestSeq), float(OriginCityMarket), Origin_ID, dayofmonth]]

# Escaladores
with open('Data/X_scaler_NO.pkl', 'br')as f:
    X_scaler = pkl.load(f)

with open('Data/y_scaler_NO.pkl', 'br')as f:
    y_scaler = pkl.load(f)

# Modelo
model = load_model('Data/regression_model_2_NO.keras')

# Escalamos X
X = X_scaler.transform(X)

# Predictor (yhat)
yhat = model.predict(X)

#Reescalamos
y = y_scaler.inverse_transform(yhat)

# Resultado
if y[0] < 0:
    st.header(f'Su vuelo se adelanta: {str(round(abs(y[0][0])))} minutos' )
else:
    st.header(f'Su vuelo se atrasa: {str(round(y[0][0]))} minutos' )

# Mapa vuelo
longit= df[df['OriginCityName']==ciudadorigen]['LONGITUDE'].unique()[0]
latit= df[df['OriginCityName']==ciudadorigen]['LATITUDE'].unique()[0]
longit_dest= df[df['DestCityName']==ciudaddest]['LONGITUDE'].unique()[0]
latit_dest= df[df['DestCityName']==ciudaddest]['LATITUDE'].unique()[0]
df_mapa = pd.DataFrame([longit,latit,longit_dest,latit_dest], columns=['longit','latit','longit_dest','latit_dest'],resetindex= True)
st.dataframe(df_mapa)

Y_RGB = [255, 255, 0, 40]
G_RGB = [56, 191, 140, 40]

# st.pydeck_chart(pdk.Deck( map_style=None, 
#                          initial_view_state=pdk.ViewState(latitude=(longit - longit_dest),
#                                                           longitude= (latit - latit_dest), 
#                                                           zoom=2.6,
#                                                           pitch=0,),
#                          layers=[pdk.Layer("ArcLayer",
#                                             data= df,
#                                             get_width= 1,
#                                             get_source_position=[longit, latit],
#                                             get_target_position=[longit_dest, latit_dest],
#                                             get_tilt=1,
#                                             get_source_color=Y_RGB,
#                                             get_target_color=G_RGB,
#                                             pickable=True,
#                                             auto_highlight=True,
#                                             )]))

CircleLayer= pdk.Layer("GreatCircleLayer",
                    data= df_mapa,
                    get_stroke_width=12,
                    get_source_position= ["longit", "latit"],
                    get_target_position= ["longit_dest", "latit_dest"],
                    get_source_color= Y_RGB,
                    get_target_color= G_RGB,
                    pickable= True,
                    auto_highlight=True
                    )

view_state = pdk.ViewState(latitude=38,
                           longitude=-98,
                           zoom=2.6,
                           pitch=50)

st.pydeck_chart(pdk.Deck(layers= CircleLayer, initial_view_state= view_state))

