import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import random
from PIL import Image

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

    # Interfaz de usuario
    st.title('Simulador de batallas de superheroes')

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






























# st.sidebar.title('Heroes')
# opciones = st.sidebar.selectbox('Selecciona:', ['Introduccion', 'Interfaz basica'])
# if opciones =='Introduccion': 
#     st.title('Bienvenidos al Superhero Simulator')
# else: 
#     st.title('Simulador de batallas de superheroes')
#     st.write('Un enfoque basado en el Machine Learning para predecir resultados en el universo de Marvel y DC')
#     st.markdown('---')
#     st.markdown('---')
#     st.header('Interfaz basica')
#     name = st.text_input('Ingresa el heroe')
#     st.write(f'Hola, {name}!')
#     intelligence = st.slider('Selecciona grado de inteligencia:', 0, 100, 50)
#     st.write(f'Tiene {intelligence} puntos de inteligencia')
#     st.markdown('---')
#     st.markdown('---')
#     st.title('Visualización del dataset')
#     st.markdown('---')
#     st.markdown('---')
#     st.header('Visualizacion del heroe')
#     imagen = Image.open(('images/spiderman.jpg'))
#     st.image(imagen, caption = 'Imagen de Spider-man')