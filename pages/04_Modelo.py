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

# imagenes redondeadas
st.markdown("""
    <style type="text/css">
    img {
    border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

#####################################################################################################################

st.header('Modelo de predicción')

st_lottie(requests.get("https://lottie.host/967d1c31-4c68-46a3-b66e-048e25590778/TIuOpUp3VG.json").json(), height=300, key="airport-Modelo")

# Cargamos las imágenes

ft_imp = Image.open("images/feature_importances.png")
f_reg = Image.open("images/f_regression.jpeg")
mutual_reg = Image.open("images/mutual_info_regression.jpeg")
model_summary = Image.open("images/model_summary.png")
model_history = Image.open("images/model_history.png")
bets_r2 = Image.open("images/best_r2.png")
val_loss = Image.open("images/val_loss.png")
val_mae = Image.open("images/val_mae.png")

# Importancia Columnas

st.subheader("Relevancia de las columnas")
st.write("Con el dataframe completo, con todas las columnas, utilizamos ```feature_importances_``` de ```RandomForestRegressor()```, y ```feature_selection``` con ```f_regression``` y ```mutual_info_regression``` para obtener las columnas más relevantes.")
st.markdown("#### Feature Importances")
st.write("```feature_importances_``` se utiliza para calcular la importancia de cada característica en el conjunto de datos. Devuelve un vector que representa la importancia de cada característica. Cuanto más alto es el valor, más importancia tiene esa característica para el modelo.")
st.image(ft_imp, use_column_width = True)
st.markdown("#### Feature Selection")
st.write("```f_regression``` mide la fuerza de la relación lineal entre una característica y la variable objetivo. Un valor alto indica una relación lineal fuerte, mientras que un valor bajo indica que la relación es estadísticamente significativa.")
st.image(f_reg, use_column_width = True)
st.write("```mutual_info_regression``` mide la cantidad de información que una característica aporta sobre la variable objetivo. Un valor alto de entropía mutua indica que la característica es muy informativa sobre la variable objetivo.")
st.image(mutual_reg, use_column_width = True)
st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

# Modelo

st.subheader("Entrenamiento del modelo")
codigo = st.expander(label = 'Código del Modelo', expanded = True)
codigo.write("""```
model = Sequential()

# Entrada
model.add(Dense(units = 128, input_shape = (X.shape[1], ), activation = "relu"))

# Capas ocultas
model.add(Dense(units = 256, activation = "relu"))
model.add(Dropout(0.2))

# model.add(Dense(units = 128, activation = "relu")) -----> # INNECESARIO
# model.add(Dropout(0.2))
# model.add(Dense(units = 64, activation = "relu")) -----> # INNECESARIO
# model.add(Dropout(0.2))

# Salida
model.add(Dense(1))

# Compilar el modelo
model.compile(optimizer = "adam", loss = "mse", metrics = ["mae"])
```""")
codigo.write("")
st.write("Utilizamos un modelo ```Sequential()``` de ```keras``` para realizar la regresión de la variable ```DepDelay``` que queremos predecir. Comprobamos que al añadir nuevas capas densas y ganar parámetros a entrenar es peor para el modelo y para sus métricas. Un único Dropout es suficiente.")
st.image(model_summary, use_column_width = True)
st.write("Ajustamos los epochs a 3 porque comprobamos que a partir del quinto epoch aumenta la pérdida; 10 serían innecesarios.")
st.write("Guardamos el modelo para posteriormente seguir entrenándolo.")
st.write("Volvemos a entrenar el modelo con 2 epochs más esperando un mejor ```r2_score```.")
st.image(model_history, use_column_width = True)

st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

# Métrica y Visualización

st.subheader("Métrica del modelo")
st.write("Después de realizar 5 epochs, en total, para entrenar el modelo, obtuvimos un ```r2_score``` de ```0.836```.")
st.write("```r2_score``` es una métrica que se utiliza para evaluar el rendimiento de un modelo de regresión.")
st.write("Mide la proporción de la varianza de los datos de destino que se explica por el modelo.")
st.write("Un valor de ```r2_score``` de 1 indica que el modelo explica toda la varianza de los datos de destino, mientras que un valor de 0 indica que el modelo no explica ninguna de la varianza.")
st.image(bets_r2, use_column_width = True)

st.subheader("Visualización de los resultados")
st.markdown("#### Loss (Mean Squared Error)")
st.write("El ```loss``` y el ```validation loss``` (```Mean Squared Error```, por defecto) son métricas clave utilizadas para evaluar el rendimiento de un modelo de ```Keras``` durante el entrenamiento. Estas métricas son esenciales para comprender cómo está aprendiendo el modelo y si está generalizando bien a datos que no ha visto durante el entrenamiento.")
masinfo1 = st.expander(label = 'Más info')
masinfo1.write("Durante el entrenamiento, es común observar que la pérdida de entrenamiento disminuye con el tiempo, ya que el modelo se ajusta mejor a los datos de entrenamiento. Sin embargo, la pérdida de validación puede tener un comportamiento diferente.")
masinfo1.write("Si tanto la pérdida de entrenamiento como la pérdida de validación disminuyen de manera similar, es un signo positivo de que el modelo está aprendiendo bien y generalizando adecuadamente.")
st.markdown("#### Mean Absolute Error")
st.write("El ```validation mae``` es crucial para evaluar la capacidad de generalización del modelo. Si el ```mae``` en el conjunto de validación es significativamente mayor que el ```mae``` de entrenamiento, puede indicar que el modelo está sobreajustando los datos de entrenamiento y no generaliza bien a datos nuevos.")
masinfo2 = st.expander(label = 'Más info')
masinfo2.write("El ```mae``` es una medida de la magnitud promedio de los errores en las predicciones del modelo. Cuanto menor sea el valor del ```mae```, mejor será el rendimiento del modelo.")
masinfo2.write("El ```validation mae``` es similar al ```mae```, pero se calcula en un conjunto de datos de validación, que consta de datos que el modelo no ha visto durante el entrenamiento.")
col1, col2 = st.columns(2)
col1.write("Loss")
col1.image(val_loss, use_column_width = True)
col2.write("Mean Absolute Error")
col2.image(val_mae, use_column_width = True)

st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

# CONFESIÓN

st.subheader('Conclusión')
st.write('Una vez realizado el modelo, al realizar la predicción, comprobamos que había dos columnas significativamente relacionadas con la variable a predecir ```DepDelay```. Estas columnas son ```WheelsOff``` y ```WheelsOn```, las cuales, idealmente, no se incluirían en el vector para entrenar al modelo ya que afectan considerablemente el rendimiento del mismo.')
st.write('Ambas columnas contienen la hora local del momento en el que el avión guarda y saca las ruedas, respectivamente.')
st.write('Si descartamos estas columnas, el modelo reduciría drásticamente su desempeño, convirtiéndolo en poco fiable. Las columnas que se pueden usar para nuestro propósito, sin contar con ```WheelsOff``` y ```WheelsOn```, no son suficientemente relevantes para entrenar un modelo fiable.')
col1, col2, col3 = st.columns([0.5, 1, 0.5])
col2.markdown("![Michael Scott GIF](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExOW4wYWI4cXltOG9lamRhNnczbzc3M2p2aTJtaXBpZ3FnMnA4dzI1NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7VHV66bRjGRSo/giphy.gif)")
st.write('Llegados a este punto, con un modelo entrenado y a medio camino de terminar el Streamlit, decidimos usar estas columnas en nuestro "Predictor". Las generamos a partir de la columna ```DepTimeBlk```, que contiene el intervalo de tiempo en el que saldría el vuelo, y la diferencia horaria entre el origen y el destino del vuelo.')
st.write("Para más información, se puede revisar el código al que hacemos referencia en el desplegable.")

confesion = st.expander(label = 'Código')

confesion.subheader('Calculamos "WheelsOff"')
confesion.write("""```
blktime_1 = datetime.datetime.strptime(str(int(franjahora.split('-')[1]) - random.randint(0, 59)), "%H%M")

blktime_2 = (datetime.datetime.strptime(str(franjahora.split('-')[1]), "%H%M") + datetime.timedelta(minutes=random.randint(0, 59)))

wheels_off = random.choice([blktime_1, blktime_2])
```""")
confesion.write("")
confesion.subheader('Calculamos "WheelsOn"')
confesion.write("""```
# DISTANCIA ENTRE ORIGEN Y DESTINO USANDO LATITUD Y LONGITUD

def rad2deg(radians):
    degrees = radians * 180 / math.pi
    return degrees

def deg2rad(degrees):
    radians = degrees * math.pi / 180
    return radians

def getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2, unit = 'kilometers'):
    
    theta = longitude1 - longitude2
    
    distance = 60 * 1.1515 * rad2deg(
        math.acos((math.sin(deg2rad(latitude1)) * math.sin(deg2rad(latitude2))) + 
                  (math.cos(deg2rad(latitude1)) * math.cos(deg2rad(latitude2)) * math.cos(deg2rad(theta)))))
    
    if unit == 'miles':
        return round(distance, 2)
    if unit == 'kilometers':
        return round(distance * 1.609344, 2)
    
distance = getDistanceBetweenPointsNew(lat1, lon1, lat2, lon2)
```""")
confesion.write("")
confesion.write("""```
# TIEMPO EN EL AIRE

velociddecrucero = 840

tiempoaire = distance/velociddecrucero

tiempotaxi = float(taxi_out)+float(df[df['Airline']==Aerolinea]['TaxiOut'].mode())

tiempovuelo = tiempoaire+(taxi_out+float(df[df['Airline']==Aerolinea]['TaxiOut'].mode()))/60

tiempodevueloreal = tiempovuelo         
```""")
confesion.write("")
confesion.write("""```
# CALCULAMOS "wheels_on" + DIFERENCIA HORARIA ENTRE ORIGEN Y DESTINO

uso_1, uso_2, uso_3, uso_4, uso_5, uso_6, uso_7 = 1, 2, 3, 4, 5, 6, 21

usos_horarios= {'Hawaii': uso_1, 'Alasca': uso_2, 'Washington': uso_3, 'Oregon': uso_3, 'Nevada': uso_3, 'California': uso_3,
                 'Montana': uso_4, 'Idaho': uso_4, 'Utah': uso_4, 'Wyoming': uso_4,'Colorado': uso_4, 'Arizona': uso_4, 'New Mexico': uso_4,
                 'North Dakota': uso_5, 'South Dakota': uso_5, 'Nebraska': uso_5, 'Kansas': uso_5, 'Oklahoma': uso_5, 'Texas': uso_5, 
                 'Minnesota': uso_5, 'Iowa': uso_5, 'Missouri': uso_5, 'Arkansas': uso_5, 'Louisiana': uso_5,'Wisconsin': uso_5,
                 'Illinois': uso_5, 'Tenessee': uso_5, 'Mississippi': uso_5, 'Alabama': uso_5, 'Michigan': uso_6, 'Indiana': uso_6, 'Kentucky': uso_6, 
                 'Georgia': uso_6, 'Florida': uso_6, 'Ohio': uso_6, 'West Virginia': uso_6, 'Virginia': uso_6, 'North Carolina': uso_6, 
                 'South Carolina': uso_6, 'Pennsylvania': uso_6, 'Delaware': uso_6, 'New York': uso_6, 'Vermont': uso_6, 'Maine': uso_6, 
                 'Maryland': uso_6, 'New Jersey': uso_6, 'New Hampshire': uso_6, 'Massachusetts': uso_6, 'Rhode Island': uso_6, 'Connecticut': uso_6, 
                 'Puerto Rico': uso_6, 'U.S. Virgin Islands': uso_6, 'U.S. Pacific Trust Territories and Possessions': uso_7}

wheels_on = wheels_off + datetime.timedelta(hours = (int(usos_horarios.get(estadoorigen)) - int(usos_horarios.get(estadodestino))), minutes = int(tiempoaire*60))

wheels_off = int(datetime.time.strftime(wheels_off.time(),'%H%M'))

wheels_on = int(datetime.time.strftime(wheels_on.time(),'%H%M'))
```""")
confesion.write("")
