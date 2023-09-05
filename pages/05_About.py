import streamlit as st
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

##### About ###############################################################

st.header('Team')
col_1, col_2, col_3 = st.columns(3)

with col_1:
    st.image('https://media.licdn.com/dms/image/D4D03AQEZNDYmDw6xbQ/profile-displayphoto-shrink_400_400/0/1687197904792?e=1697673600&v=beta&t=punW-e_QYI9KavGDk8XWqvzckRiuNT9yzGn3d1BDfvY',
            width= 200,)
    st.subheader('Germán Fernández')
    # if col_1.button(label = 'Linkedin Germán', use_container_width= True, ):
    #     webbrowser.open_new_tab('https://www.linkedin.com/in/german-fernandez-corrales/')
    link = '[LinkedIn](https://www.linkedin.com/in/german-fernandez-corrales/)'
    github = '[GitHub](https://github.com/f00dez)'
    st.markdown(link, unsafe_allow_html=True)
    st.markdown(github,unsafe_allow_html=True)

with col_2:
    st.image('https://media.licdn.com/dms/image/D4D03AQHaDZ_9s0lxmw/profile-displayphoto-shrink_400_400/0/1686648949588?e=1697673600&v=beta&t=ICAUdXK_vGrLMuzRQVqEmNW5rmQai1INiEzbSkI9LLY',
            width= 200,)
    st.subheader('Miguel Nieto')
    # if col_2.button(label = 'Linkedin Miguel', use_container_width= True, ):
    #     webbrowser.open_new_tab('https://www.linkedin.com/in/miguel-nieto-p/')
    link = '[LinkedIn](https://www.linkedin.com/in/miguel-nieto-p/)'
    github = '[GitHub](https://github.com/miguelnietop)'
    st.markdown(link, unsafe_allow_html=True)
    st.markdown(github,unsafe_allow_html=True)

with col_3:
    st.image('https://media.licdn.com/dms/image/D4D03AQGJSyQ4v4QVFw/profile-displayphoto-shrink_800_800/0/1687511681863?e=2147483647&v=beta&t=kJ_dBjX9dpThCF6CfqgSGo8R-9j8hNSJTbXQDewXHYU',
            width= 200,)
    st.subheader('Sergio Soler')
    # if col_3.button(label = 'Linkedin Sergio', use_container_width= True, ):
    #     webbrowser.open_new_tab('https://www.linkedin.com/in/sergiosolergarcia/')
    link = '[LinkedIn](https://www.linkedin.com/in/sergiosolergarcia/)'
    github = '[GitHub](https://github.com/Solerypunto)'
    st.markdown(link, unsafe_allow_html=True)
    st.markdown(github,unsafe_allow_html=True)

# imagenes redondas
st.markdown("""
    <style type="text/css">
    img {
    border-radius: 10000px;
    }
    </style>
    """, unsafe_allow_html=True)