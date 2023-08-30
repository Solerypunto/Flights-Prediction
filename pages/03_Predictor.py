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

##### Predictor ###############################################################

st_lottie(requests.get("https://lottie.host/e94e8eb0-c1ed-41c3-b6da-b22a5104f594/4sofR9rtYA.json").json(), height=200, key="paperplane")

st.header('¿Cual es tu vuelo?')

fechavuelo = st.date_input("¿Cuando es tu vuelo?", datetime.date(2023, 9, 10))
st.write('Tu vuelo sale el:', fechavuelo)

horavuelo = st.date_input("¿Cuando es tu vuelo?", datetime.time())
st.write('Tu vuelo sale a las:', horavuelo)