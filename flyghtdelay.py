import streamlit as st
import webbrowser
import pandas as pd
import numpy as np
import pydeck as pdk
# import plotly.figure_factory as ff
import plotly.express as px

path = "Data/prueba_streamlit.csv"
df_all = pd.read_csv(path, sep=',', header= 0, )

##### Configuracion de la página ###############################################################

st.set_page_config(page_title= 'Flight delay predictor',
                   page_icon= ':airplane_departure:',
                   initial_sidebar_state= 'collapsed', layout= 'wide',)

##### Cuerpo de la página ###############################################################
def main():

    st.title('FLIGHT DELAY PREDICTOR')

    home, eda, predictor, about = st.tabs(['Home', 'EDA', 'Predictor', 'About'])

##### Home ###############################################################

    with home:
        st.header('Esto es un :orange[header] de prueba')
        st.subheader('Esto es un subheader pa probá a ve como queda esto')
        st. write('''Aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, 
                  aquí va una intro del proyecto para que la gente se entere de que va, ''')
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#ffe100;" /> """, unsafe_allow_html=True)

##### EDA ###############################################################

# Mapa 3D
# https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart 

    with eda:
        st.subheader('Gráfica 1')

        col_a, col_b = st.columns([1,2])

        col_a.dataframe(df_all.groupby("Origin").agg(**{'max':('DepDelay','max'),'mean':('DepDelay','mean')}).reset_index())

        col_b.write( 'Para ver como se relaciona la media de retraso con el aeropuerto de origen. Cuanto mas grande sea el circulo mas alejado está de la media')
                                                        
        dfimpresionante = df_all.groupby("Origin").agg(**{'max':('DepDelay','max'),'mean':('DepDelay','mean')}).reset_index()

        fig = px.scatter(dfimpresionante,
                 x = 'Origin',
                 y = "mean",
                 size = dfimpresionante['max']+20,
                 color = 'Origin')

        st.plotly_chart(fig, theme='streamlit', use_container_width=True)
        

        st.markdown('''---''')

        st.subheader('Gráfica 2')
        df = px.data.iris()
        fig = px.scatter(
            df,
            x="sepal_width",
            y="sepal_length",
            color="sepal_length",
            color_continuous_scale=[[0, 'grey'], [1.0, 'rgb(255, 255, 0)']],)

        st.plotly_chart(fig, theme='streamlit', use_container_width=True)


##### About ###############################################################

    with about:
        st.header('Team')
        col_1, col_2, col_3 = st.columns(3)

        with col_1:
            st.image('https://media.licdn.com/dms/image/D4D03AQEZNDYmDw6xbQ/profile-displayphoto-shrink_400_400/0/1687197904792?e=1697673600&v=beta&t=punW-e_QYI9KavGDk8XWqvzckRiuNT9yzGn3d1BDfvY',
                    width= 200,)
            st.subheader('Germán Fernandez')
            # if col_1.button(label = 'Linkedin Germán', use_container_width= True, ):
            #     webbrowser.open_new_tab('https://www.linkedin.com/in/german-fernandez-corrales/')
            link = '[Linkedin](https://www.linkedin.com/in/german-fernandez-corrales/)'
            st.markdown(link, unsafe_allow_html=True)

        with col_2:
            st.image('https://media.licdn.com/dms/image/D4D03AQHaDZ_9s0lxmw/profile-displayphoto-shrink_400_400/0/1686648949588?e=1697673600&v=beta&t=ICAUdXK_vGrLMuzRQVqEmNW5rmQai1INiEzbSkI9LLY',
                    width= 200,)
            st.subheader('Miguel Nieto')
            # if col_2.button(label = 'Linkedin Miguel', use_container_width= True, ):
            #     webbrowser.open_new_tab('https://www.linkedin.com/in/miguel-nieto-p/')
            link = '[Linkedin](https://www.linkedin.com/in/miguel-nieto-p/)'
            st.markdown(link, unsafe_allow_html=True)
        
        with col_3:
            st.image('https://media.licdn.com/dms/image/D4D03AQGJSyQ4v4QVFw/profile-displayphoto-shrink_800_800/0/1687511681863?e=2147483647&v=beta&t=kJ_dBjX9dpThCF6CfqgSGo8R-9j8hNSJTbXQDewXHYU',
                    width= 200,)
            st.subheader('Sergio Soler')
            # if col_3.button(label = 'Linkedin Sergio', use_container_width= True, ):
            #     webbrowser.open_new_tab('https://www.linkedin.com/in/sergiosolergarcia/')
            link = '[Linkedin](https://www.linkedin.com/in/sergiosolergarcia/)'
            st.markdown(link, unsafe_allow_html=True)

    pass

if __name__ == '__main__':
    main()

## CSS ###############################################################   
# tipografia   
streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Space Mono', sans-serif;
            font-weigth: 900;
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)
# imagenes redondas
st.markdown("""
            <style type="text/css">
            img {
            border-radius: 10000px;
            }
            </style>
            """, unsafe_allow_html=True)