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

##### About ###############################################################

st.header('Team')
col_1, col_2, col_3 = st.columns(3)

with col_1:
    st.image('https://media.licdn.com/dms/image/D4D03AQEZNDYmDw6xbQ/profile-displayphoto-shrink_400_400/0/1687197904792?e=1697673600&v=beta&t=punW-e_QYI9KavGDk8XWqvzckRiuNT9yzGn3d1BDfvY',
            width= 200,)
    st.subheader('Germán Fernandez')
    # if col_1.button(label = 'Linkedin Germán', use_container_width= True, ):
    #     webbrowser.open_new_tab('https://www.linkedin.com/in/german-fernandez-corrales/')
    link = '[LinkedIn](https://www.linkedin.com/in/german-fernandez-corrales/)'
    st.markdown(link, unsafe_allow_html=True)

with col_2:
    st.image('https://media.licdn.com/dms/image/D4D03AQHaDZ_9s0lxmw/profile-displayphoto-shrink_400_400/0/1686648949588?e=1697673600&v=beta&t=ICAUdXK_vGrLMuzRQVqEmNW5rmQai1INiEzbSkI9LLY',
            width= 200,)
    st.subheader('Miguel Nieto')
    # if col_2.button(label = 'Linkedin Miguel', use_container_width= True, ):
    #     webbrowser.open_new_tab('https://www.linkedin.com/in/miguel-nieto-p/')
    link = '[LinkedIn](https://www.linkedin.com/in/miguel-nieto-p/)'
    st.markdown(link, unsafe_allow_html=True)

with col_3:
    st.image('https://media.licdn.com/dms/image/D4D03AQGJSyQ4v4QVFw/profile-displayphoto-shrink_800_800/0/1687511681863?e=2147483647&v=beta&t=kJ_dBjX9dpThCF6CfqgSGo8R-9j8hNSJTbXQDewXHYU',
            width= 200,)
    st.subheader('Sergio Soler')
    # if col_3.button(label = 'Linkedin Sergio', use_container_width= True, ):
    #     webbrowser.open_new_tab('https://www.linkedin.com/in/sergiosolergarcia/')
    link = '[LinkedIn](https://www.linkedin.com/in/sergiosolergarcia/)'
    st.markdown(link, unsafe_allow_html=True)

# imagenes redondas
st.markdown("""
    <style type="text/css">
    img {
    border-radius: 10000px;
    }
    </style>
    """, unsafe_allow_html=True)