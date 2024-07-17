import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import random
from sklearn.preprocessing import LabelEncoder
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo del modelo
rf_mod = './best_rf_model.pkl'

# Aplicar estilo CSS para el fondo y el texto
def set_background(image_file):
    with open(image_file, "rb") as image_file:
        image_data = image_file.read()
    encoded_image = base64.b64encode(image_data).decode()

    background_style = f"""
    <style>
    .stApp {{
        background-color: #323236;  /* Color de fondo para toda la aplicación */
    }}
    .portada {{
        background-image: url(data:image/jpeg;base64,{encoded_image});
        background-size: cover;
        background-attachment: fixed;
        padding: 50px;
        text-align: center;
    }}
    .title {{
        color: #1f77b4;  /* Color del título */
        text-align: center;
    }}
    .subtitle {{
        color: #ff7f0e;  /* Color del subtítulo */
        text-align: center;
    }}
    .text {{
        color: #2ca02c;  /* Color del texto */
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Llama a la función set_background con la ruta de tu imagen
set_background('./assets/portada_app.jpg')

# Resto del código de tu aplicación
if not os.path.isfile(rf_mod):
    st.error(f"El archivo del modelo '{rf_mod}' no se encuentra en el directorio especificado.")
else:
    # Cargar el modelo
    model = joblib.load(rf_mod)

    # Cargar el dataset
    df_pred_battles = pd.read_csv('./data/MCU_DC_final_limpio.csv')
    
    # Label Encoder Heroe 1 y Heroe 2
    label_encoder = LabelEncoder()
    df_pred_battles['Heroe_1_encoded'] = label_encoder.fit_transform(df_pred_battles['Heroe 1'])
    df_pred_battles['Heroe_2_encoded'] = label_encoder.fit_transform(df_pred_battles['Heroe 2'])

    # Función para simular una batalla
    def simular_batalla(features, model, heroe1, heroe2):
        features_df = pd.DataFrame(features, index=[0])
        prediction = model.predict(features_df)
        if prediction[0] == 1:
            return f"{heroe1} gana"
        else:
            return f"{heroe2} gana"

    # Función para obtener atributos del héroe
    def obtener_atributos(heroe, heroe_col):
        atributos = df_pred_battles[df_pred_battles[heroe_col] == heroe].iloc[0]
        return {
            'intelligence': atributos[f'Intelligence_{heroe_col[-1]}'],
            'strength': atributos[f'Strength_{heroe_col[-1]}'],
            'speed': atributos[f'Speed_{heroe_col[-1]}'],
            'durability': atributos[f'Durability_{heroe_col[-1]}'],
            'power': atributos[f'Power_{heroe_col[-1]}'],
            'combat': atributos[f'Combat_{heroe_col[-1]}']
        }

    # Función para mostrar la imagen del héroe
    def mostrar_imagen_heroe(heroe, col):
        image_path = f'./images/{heroe}.jpeg'
        if os.path.isfile(image_path):
            imagen = Image.open(image_path)
            st.image(imagen, caption=f'Imagen de {heroe}', use_column_width=True)
        else:
            st.warning(f'No se encontró la imagen para {heroe}')

    if 'entrar' not in st.session_state:
        st.session_state.entrar = False

    if not st.session_state.entrar:
        st.markdown('<div class="portada">', unsafe_allow_html=True)
        st.markdown('<h1 class="title">⚡ Superhero Battle Arena: The Ultimate Showdown ⚡</h1>', unsafe_allow_html=True)
        st.markdown("""
        <p class="text">
        **¡Bienvenido a la Superhero Battle Arena!** Sumérgete en el emocionante mundo de los superhéroes con esta innovadora aplicación.
        Utilizamos técnicas avanzadas de Machine Learning y un completo Análisis Exploratorio de Datos (EDA) para ofrecerte una experiencia única. 
        Explora datos detallados de tus héroes favoritos y simula batallas épicas para descubrir quién se alzará con la victoria en el combate definitivo.
        ¡Prepárate para vivir la ciencia detrás de cada enfrentamiento y disfruta de la emoción de la Superhero Battle Arena!
        </p>
        """, unsafe_allow_html=True)
        if st.button('Entrar'):
            st.session_state.entrar = True
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<h1 class="title">⚡ Superhero Battle Arena: The Ultimate Showdown ⚡</h1>', unsafe_allow_html=True)

        # Crear pestañas
        tabs = st.tabs(["EDA", "Simulador"])


        # Pestaña EDA
        with tabs[0]:
            st.markdown('<h2 class="subtitle">Análisis Exploratorio de Datos (EDA)</h2>', unsafe_allow_html=True)

            # Mostrar el dataframe
            st.write('<p class="text">Datos de las batallas:</p>', unsafe_allow_html=True)
            st.dataframe(df_pred_battles)

            # Análisis de los targets
            st.write('<p class="text">Distribución de los targets:</p>', unsafe_allow_html=True)
            targets = ['Resultado_Combinado', 'Resultado_Peliculas', 'Resultado_Comics']  
            fig, axes = plt.subplots(1, len(targets), figsize=(16, 5))
            for i, target in enumerate(targets):
                sns.countplot(x=df_pred_battles[target], ax=axes[i])
                axes[i].set_title(f'Distribución de {target}')
                axes[i].set_xlabel(target)
                axes[i].set_ylabel('Frecuencia')
            plt.tight_layout()
            st.pyplot(fig)

            # Visualización de variables numéricas
            st.write('<p class="text">Visualización de variables numéricas:</p>', unsafe_allow_html=True)
            num_var = ['Intelligence_1', 'Strength_1', 'Speed_1', 'Durability_1', 'Power_1', 'Combat_1',
                    'Intelligence_2', 'Strength_2', 'Speed_2', 'Durability_2', 'Power_2', 'Combat_2']

            plt.figure(figsize=(16, 40))
            for i, column in enumerate(num_var, 1):
                plt.subplot(len(num_var), 2, 2*i-1)
                sns.histplot(df_pred_battles[column], bins=20, kde=True)
                plt.title(f'Histograma de {column}')
                plt.xlabel(column)
                plt.ylabel('Frecuencia')

                plt.subplot(len(num_var), 2, 2*i)
                sns.boxplot(y=df_pred_battles[column])
                plt.title(f'Boxplot de {column}')
                plt.ylabel(column)

            plt.tight_layout()
            st.pyplot(plt.gcf())

            # Plot bar plot for 'Heroe 1'
            st.write('<p class="text">Distribución de Heroe:</p>', unsafe_allow_html=True)
            plt.figure(figsize=(14, 6))
            serie = df_pred_battles['Heroe 1'].value_counts()
            sns.barplot(x=serie.index, y=serie, palette='viridis')
            plt.title('Distribución de Heroe 1')
            plt.xlabel('Heroe')
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(plt.gcf())



            # Definir los atributos y las variables objetivo
            atributos = ['Intelligence_1', 'Strength_1', 'Speed_1', 'Durability_1', 'Power_1', 'Combat_1',
                        'Intelligence_2', 'Strength_2', 'Speed_2', 'Durability_2', 'Power_2', 'Combat_2']
            targets_comics = 'Resultado_Comics'
            targets_peliculas = 'Resultado_Peliculas'

            # Separar los datos en ganadores y perdedores para cómics y películas
            ganadores_comics = df_comics_pelis[df_comics_pelis[targets_comics] == 1]
            perdedores_comics = df_comics_pelis[df_comics_pelis[targets_comics] == 2]

            ganadores_peliculas = df_comics_pelis[df_comics_pelis[targets_peliculas] == 1]
            perdedores_peliculas = df_comics_pelis[df_comics_pelis[targets_peliculas] == 2]

            # Visualización de atributos para cómics
            def visualizar_atributos_bar(ganadores, perdedores, atributos, titulo):
                fig, axes = plt.subplots(6, 2, figsize=(10, 20))
                fig.suptitle(f'Comparación de Atributos entre Ganadores y Perdedores ({titulo})', fontsize=16)
                
                for i, atributo in enumerate(atributos):
                    mean_ganadores = ganadores[atributo].mean()
                    mean_perdedores = perdedores[atributo].mean()
                    
                    sns.barplot(x=['Ganadores', 'Perdedores'], y=[mean_ganadores, mean_perdedores], ax=axes[i//2, i%2])
                    axes[i//2, i%2].set_title(f'{atributo}')
                    axes[i//2, i%2].set_ylabel('Media')
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                st.pyplot(fig)  # Utiliza st.pyplot en Streamlit en lugar de plt.show() para mostrar gráficos

            # Visualizar atributos para cómics
            visualizar_atributos_bar(ganadores_comics, perdedores_comics, atributos, 'Cómics')

            # Visualizar atributos para películas
            visualizar_atributos_bar(ganadores_peliculas, perdedores_peliculas, atributos, 'Películas')














        # Pestaña Simulador
        with tabs[1]:
            st.markdown('<h2 class="subtitle">Simulador de batallas de superhéroes</h2>', unsafe_allow_html=True)

            # Selección de héroes
            heroe1 = st.selectbox('Selecciona el Héroe 1', ['Aleatorio'] + list(df_pred_battles['Heroe 1'].unique()))
            heroe2 = st.selectbox('Selecciona el Héroe 2', ['Aleatorio'] + list(df_pred_battles['Heroe 2'].unique()))

            # Obtener atributos del Héroe 1
            if heroe1 == 'Aleatorio':
                st.markdown('<h3 class="subtitle">Atributos de Héroe 1</h3>', unsafe_allow_html=True)
                intelligence_1 = st.slider('Inteligencia', 0, 100, 50, key='intelligence_1')
                strength_1 = st.slider('Fuerza', 0, 100, 50, key='strength_1')
                speed_1 = st.slider('Velocidad', 0, 100, 50, key='speed_1')
                durability_1 = st.slider('Durabilidad', 0, 100, 50, key='durability_1')
                power_1 = st.slider('Poder', 0, 100, 50, key='power_1')
                combat_1 = st.slider('Combate', 0, 100, 50, key='combat_1')
                heroe1_encoded = random.randint(0, 500)  # Asignar un valor aleatorio
            else:
                st.markdown(f'<h3 class="subtitle">Atributos de {heroe1}</h3>', unsafe_allow_html=True)
                atributos_1 = obtener_atributos(heroe1, 'Heroe 1')
                intelligence_1 = st.slider('Inteligencia', 0, 100, atributos_1['intelligence'], key='intelligence_1', disabled=True)
                strength_1 = st.slider('Fuerza', 0, 100, atributos_1['strength'], key='strength_1', disabled=True)
                speed_1 = st.slider('Velocidad', 0, 100, atributos_1['speed'], key='speed_1', disabled=True)
                durability_1 = st.slider('Durabilidad', 0, 100, atributos_1['durability'], key='durability_1', disabled=True)
                power_1 = st.slider('Poder', 0, 100, atributos_1['power'], key='power_1', disabled=True)
                combat_1 = st.slider('Combate', 0, 100, atributos_1['combat'], key='combat_1', disabled=True)
                heroe1_encoded = df_pred_battles.loc[df_pred_battles['Heroe 1'] == heroe1, 'Heroe_1_encoded'].values[0]
                mostrar_imagen_heroe(heroe1, 'Heroe 1')

            # Obtener atributos del Héroe 2
            if heroe2 == 'Aleatorio':
                st.markdown('<h3 class="subtitle">Atributos de Héroe 2</h3>', unsafe_allow_html=True)
                intelligence_2 = st.slider('Inteligencia', 0, 100, 50, key='intelligence_2')
                strength_2 = st.slider('Fuerza', 0, 100, 50, key='strength_2')
                speed_2 = st.slider('Velocidad', 0, 100, 50, key='speed_2')
                durability_2 = st.slider('Durabilidad', 0, 100, 50, key='durability_2')
                power_2 = st.slider('Poder', 0, 100, 50, key='power_2')
                combat_2 = st.slider('Combate', 0, 100, 50, key='combat_2')
                heroe2_encoded = random.randint(0, 500)  # Asignar un valor aleatorio
            else:
                st.markdown(f'<h3 class="subtitle">Atributos de {heroe2}</h3>', unsafe_allow_html=True)
                atributos_2 = obtener_atributos(heroe2, 'Heroe 2')
                intelligence_2 = st.slider('Inteligencia', 0, 100, atributos_2['intelligence'], key='intelligence_2', disabled=True)
                strength_2 = st.slider('Fuerza', 0, 100, atributos_2['strength'], key='strength_2', disabled=True)
                speed_2 = st.slider('Velocidad', 0, 100, atributos_2['speed'], key='speed_2', disabled=True)
                durability_2 = st.slider('Durabilidad', 0, 100, atributos_2['durability'], key='durability_2', disabled=True)
                power_2 = st.slider('Poder', 0, 100, atributos_2['power'], key='power_2', disabled=True)
                combat_2 = st.slider('Combate', 0, 100, atributos_2['combat'], key='combat_2', disabled=True)
                heroe2_encoded = df_pred_battles.loc[df_pred_battles['Heroe 2'] == heroe2, 'Heroe_2_encoded'].values[0]
                mostrar_imagen_heroe(heroe2, 'Heroe 2')

            # Botón para iniciar el combate
            if st.button('Iniciar Combate'):
                nuevo_combate = {
                    'Heroe_1_encoded': heroe1_encoded,
                    'Intelligence_1': intelligence_1,
                    'Strength_1': strength_1,
                    'Speed_1': speed_1,
                    'Durability_1': durability_1,
                    'Power_1': power_1,
                    'Combat_1': combat_1,
                    'Heroe_2_encoded': heroe2_encoded,
                    'Intelligence_2': intelligence_2,
                    'Strength_2': strength_2,
                    'Speed_2': speed_2,
                    'Durability_2': durability_2,
                    'Power_2': power_2,
                    'Combat_2': combat_2
                }

                # Realizar la simulación
                resultado = simular_batalla(nuevo_combate, model, heroe1, heroe2)
                st.write(f'El resultado del combate es: {resultado}')
