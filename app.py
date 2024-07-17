import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo del modelo
rf_mod = './best_rf_model.pkl'

# Verificar si el archivo del modelo existe
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
    def simular_batalla(features, model):
        features_df = pd.DataFrame(features, index=[0])
        prediction = model.predict(features_df)
        if prediction[0] == 1:
            return "Heroe 1 gana"
        else:
            return "Heroe 2 gana"

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

    # Aplicar estilo CSS para centrar el contenido
    st.markdown(
        """
        <style>
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 80vh;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if 'entrar' not in st.session_state:
        st.session_state.entrar = False

    if not st.session_state.entrar:
        with st.container():
            st.markdown('<div class="centered">', unsafe_allow_html=True)
            st.title('⚡ Superhero Battle Arena: The Ultimate Showdown ⚡')
            st.write("""
            **¡Bienvenido a la Superhero Battle Arena!** Sumérgete en el emocionante mundo de los superhéroes con esta innovadora aplicación.
            Utilizamos técnicas avanzadas de Machine Learning y un completo Análisis Exploratorio de Datos (EDA) para ofrecerte una experiencia única. 
            Explora datos detallados de tus héroes favoritos y simula batallas épicas para descubrir quién se alzará con la victoria en el combate definitivo.
            ¡Prepárate para vivir la ciencia detrás de cada enfrentamiento y disfruta de la emoción de la Superhero Battle Arena!
            """)
            if st.button('Entrar'):
                st.session_state.entrar = True
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.title('⚡ Superhero Battle Arena: The Ultimate Showdown ⚡')

        # Crear pestañas
        tabs = st.tabs(["EDA", "Simulador"])

        # Pestaña EDA
        with tabs[0]:
            st.title('Análisis Exploratorio de Datos (EDA)')

            # Selector para el tipo de análisis
            analisis_tipo = st.selectbox('Seleccione el tipo de análisis', ['Análisis de targets', 'Analisis univariante', 'Hipotesis'])

            if analisis_tipo == 'Análisis de targets':
                st.write('Seleccione el target a analizar:')
                target_analisis = st.selectbox('Target', ['Resultado_Combinado', 'Resultado_Peliculas', 'Resultado_Comics'])

                # Mostrar el dataframe filtrado por el target seleccionado
                st.write(f'Datos de las batallas filtrado por {target_analisis}:')
                st.dataframe(df_pred_battles[[target_analisis]])

                # Ejemplo de gráficos
                st.write(f'Distribución de {target_analisis}:')
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                sns.countplot(x=target_analisis, data=df_pred_battles, ax=axes[0])
                axes[0].set_title(f'Distribución de {target_analisis}')
                axes[0].set_xlabel(target_analisis)
                axes[0].set_ylabel('Frecuencia')

                sns.histplot(df_pred_battles[target_analisis], bins=20, kde=True, ax=axes[1])
                axes[1].set_title(f'Histograma de {target_analisis}')
                axes[1].set_xlabel(target_analisis)
                axes[1].set_ylabel('Frecuencia')

                plt.tight_layout()
                st.pyplot(fig)

            elif analisis_tipo == 'Hipotesis':
                st.write('Seleccione la hipótesis a analizar:')
                hipotesis_analisis = st.selectbox('Hipótesis', ['Atributos', 'Fuerza vs Poder', 'Mas fuerte'])

                # Aquí puedes agregar el código para cada hipótesis que deseas analizar
                if hipotesis_analisis == 'Atributos':
                    st.write('Análisis de atributos...')
                elif hipotesis_analisis == 'Fuerza vs Poder':
                    st.write('Comparación de fuerza vs poder...')
                elif hipotesis_analisis == 'Mas fuerte':
                    st.write('Análisis del más fuerte...')

        # Pestaña Simulador
        with tabs[1]:
            st.title('Simulador de batallas de superhéroes')

            # Selección de héroes
            heroe1 = st.selectbox('Selecciona el Héroe 1', ['Aleatorio'] + list(df_pred_battles['Heroe 1'].unique()))
            heroe2 = st.selectbox('Selecciona el Héroe 2', ['Aleatorio'] + list(df_pred_battles['Heroe 2'].unique()))

            # Obtener atributos del Héroe 1
            if heroe1 == 'Aleatorio':
                st.header('Atributos de Héroe 1')
                intelligence_1 = st.slider('Inteligencia', 0, 100, 50, key='intelligence_1')
                strength_1 = st.slider('Fuerza', 0, 100, 50, key='strength_1')
                speed_1 = st.slider('Velocidad', 0, 100, 50, key='speed_1')
                durability_1 = st.slider('Durabilidad', 0, 100, 50, key='durability_1')
                power_1 = st.slider('Poder', 0, 100, 50, key='power_1')
                combat_1 = st.slider('Combate', 0, 100, 50, key='combat_1')
                heroe1_encoded = random.randint(0, 500)  # Asignar un valor aleatorio
            else:
                st.header(f'Atributos de {heroe1}')
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
                st.header('Atributos de Héroe 2')
                intelligence_2 = st.slider('Inteligencia', 0, 100, 50, key='intelligence_2')
                strength_2 = st.slider('Fuerza', 0, 100, 50, key='strength_2')
                speed_2 = st.slider('Velocidad', 0, 100, 50, key='speed_2')
                durability_2 = st.slider('Durabilidad', 0, 100, 50, key='durability_2')
                power_2 = st.slider('Poder', 0, 100, 50, key='power_2')
                combat_2 = st.slider('Combate', 0, 100, 50, key='combat_2')
                heroe2_encoded = random.randint(0, 500)  # Asignar un valor aleatorio
            else:
                st.header(f'Atributos de {heroe2}')
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
                resultado = simular_batalla(nuevo_combate, model)
                st.write(f'El resultado del combate es: {resultado}')
