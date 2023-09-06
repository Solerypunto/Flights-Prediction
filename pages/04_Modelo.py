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
    border-radius: 15px;
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
st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

# Modelo

st.subheader("Entrenamiento del modelo")
código = st.expander(label = 'Código del Modelo', expanded = True)
código.write("""```
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
st.write("Utilizamos un modelo ```Sequential()``` de ```keras``` para realizar la regresión de la variable ```DepDelay``` que queremos predecir. Comprobamos que al añadir nuevas capas densas y ganar parámetros a entrenar es peor para el modelo y para sus métricas. Un único Dropout es suficiente.")
st.image(model_summary, use_column_width = True)
st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

# Métrica y Visualización

st.subheader("Métrica del modelo")
st.write("")
st.image(bets_r2, use_column_width = True)

st.subheader("Visualización de la pérdida")
st.write("")
st.image(val_loss, use_column_width = True)
st.image(val_mae, use_column_width = True)

st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)
