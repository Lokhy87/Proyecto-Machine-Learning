import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy import stats

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
    df_comics_pelis = pd.read_csv('./data/MCU_DC_final_limpio.csv')

    # Label Encoder Heroe 1 y Heroe 2
    label_encoder = LabelEncoder()
    df_pred_battles['Heroe_1_encoded'] = label_encoder.fit_transform(df_pred_battles['Heroe 1'])
    df_pred_battles['Heroe_2_encoded'] = label_encoder.fit_transform(df_pred_battles['Heroe 2'])

    # Función para simular una batalla
    def simular_batalla(features, model):
        features_df = pd.DataFrame(features, index=[0])
        prediction = model.predict(features_df)
        if prediction[0] == 1:
            return "Heroe 1"
        else:
            return "Heroe 2"

    # Función para obtener atributos del héroe
    def obtener_atributos(heroe, heroe_col):
        if heroe == "Aleatorio":
            return {
                'intelligence': random.randint(0, 100),
                'strength': random.randint(0, 100),
                'speed': random.randint(0, 100),
                'durability': random.randint(0, 100),
                'power': random.randint(0, 100),
                'combat': random.randint(0, 100)
            }
        else:
            atributos = df_pred_battles[df_pred_battles[heroe_col] == heroe].iloc[0]
            col_suffix = heroe_col.split()[-1]  # Obtener el sufijo de la columna ('1' o '2')
            return {
                'intelligence': atributos[f'Intelligence_{col_suffix}'],
                'strength': atributos[f'Strength_{col_suffix}'],
                'speed': atributos[f'Speed_{col_suffix}'],
                'durability': atributos[f'Durability_{col_suffix}'],
                'power': atributos[f'Power_{col_suffix}'],
                'combat': atributos[f'Combat_{col_suffix}']
            }

    # Función para mostrar la imagen del héroe con tamaño fijo
    def mostrar_imagen_heroe(heroe, col, width=200, height=200):
        if heroe == "Aleatorio":
            image_path = './images/Aleatorio.jpeg'
        else:
            image_path = f'./images/{heroe}.jpeg'
        if os.path.isfile(image_path):
            imagen = Image.open(image_path)
            imagen = imagen.resize((width, height))  # Redimensionar la imagen
            st.image(imagen, caption=f'Imagen de {heroe}', use_column_width=False)
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
        num_var = [
            'Intelligence_1', 'Intelligence_2', 'Strength_1', 'Strength_2', 
            'Speed_1', 'Speed_2', 'Durability_1', 'Durability_2', 
            'Power_1', 'Power_2', 'Combat_1', 'Combat_2', 
            'Tier_1', 'Tier_2', 'Tier2_1', 'Tier2_2'
        ]

        # Definir los atributos y las variables objetivo
        atributos = [
            'Intelligence_1', 'Strength_1', 'Speed_1', 'Durability_1', 'Power_1', 'Combat_1',
            'Intelligence_2', 'Strength_2', 'Speed_2', 'Durability_2', 'Power_2', 'Combat_2'
        ]
        targets_comics = 'Resultado_Comics'
        targets_peliculas = 'Resultado_Peliculas'

        # Separar los datos en ganadores y perdedores para películas
        ganadores_peliculas = df_pred_battles[df_pred_battles[targets_peliculas] == 1]
        perdedores_peliculas = df_pred_battles[df_pred_battles[targets_peliculas] == 2]
    
        # Función para visualizar atributos entre ganadores y perdedores
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
            st.pyplot(fig)  # Renderizar el gráfico en Streamlit

        with tabs[0]:
            st.title('Análisis Exploratorio de Datos (EDA)')

            # Selector para el tipo de análisis
            analisis_tipo = st.selectbox('Seleccione el tipo de análisis', ['Análisis de targets', 'Analisis', 'Hipotesis'])

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

            elif analisis_tipo == 'Analisis':
                st.write('Seleccione el tipo de análisis:')
                tipo_analisis = st.selectbox('Tipo de análisis', ['Numerico', 'Categorico'])

                if tipo_analisis == 'Numerico':
                    st.write('Análisis numérico...')
                    
                    # Mostrar estadísticas descriptivas de las variables numéricas
                    st.write('Estadísticas descriptivas de las variables numéricas:')
                    st.write(df_pred_battles[num_var].describe())

                    # Ejemplo de gráficos para variables numéricas
                    st.write('Ejemplo de gráficos para variables numéricas:')

                    # Definir cuántos gráficos quieres por fila
                    graficos_por_fila = 3
                    num_vars = len(num_var)

                    # Iterar sobre las variables numéricas y crear los gráficos
                    for i in range(0, num_vars, graficos_por_fila):
                        fig, axes = plt.subplots(1, graficos_por_fila, figsize=(18, 6))
                        
                        for j, var in enumerate(num_var[i:i + graficos_por_fila]):
                            sns.histplot(df_pred_battles[var], bins=20, kde=True, ax=axes[j])
                            axes[j].set_title(f'Histograma de {var}')
                            axes[j].set_xlabel(var)
                            axes[j].set_ylabel('Frecuencia')
                        
                        plt.tight_layout()
                        st.pyplot(fig)

                elif tipo_analisis == 'Categorico':
                    st.write('Seleccione la variable categórica a analizar:')
                    variable_categorica = st.selectbox('Variable categórica', ['Universe_1', 'Universe_2'])

                    # Generar las series de valores contados
                    serie = df_pred_battles[variable_categorica].value_counts()

                    # Calcular los porcentajes
                    total = serie.sum()
                    percentages = (serie / total) * 100

                    # Crear el gráfico de barras
                    plt.figure(figsize=(12, 8))

                    # Gráfico de barras para la variable seleccionada
                    sns.barplot(x=percentages.index, y=percentages, palette='viridis')
                    plt.title(f'Distribución de {variable_categorica}')
                    plt.xlabel(variable_categorica)
                    plt.ylabel('Porcentaje')
                    plt.xticks(rotation=60)

                    # Añadir etiquetas de porcentaje encima de las barras
                    for index, value in enumerate(percentages):
                        plt.text(index, value + 0.5, f'{value:.1f}%', ha='center')

                    plt.tight_layout()
                    st.pyplot(plt.gcf())

            elif analisis_tipo == 'Hipotesis':
                st.write('Seleccione la hipótesis a analizar:')
                hipotesis_analisis = st.selectbox('Hipótesis', ['Atributos de ganadores y perdedores', 'Distribucion comics vs peliculas', 'Fuerza y Poder relacionados con la victoria', 'Similitud heroes del mismo universo'])

                # Aquí puedes agregar el código para cada hipótesis que deseas analizar
                if hipotesis_analisis == 'Atributos de ganadores y perdedores':
                    st.write('Análisis de atributos entre ganadores y perdedores')
                    st.subheader('Comparación de atributos entre ganadores y perdedores')

                    # Cargar datos para cómics (asegurando que estén cargados antes de usarlos)
                    df_comics_pelis = pd.read_csv('./data/MCU_DC_final_limpio.csv')

                    # Separar los datos en ganadores y perdedores para cómics
                    ganadores_comics = df_comics_pelis[df_comics_pelis['Resultado_Comics'] == 1]
                    perdedores_comics = df_comics_pelis[df_comics_pelis['Resultado_Comics'] == 2]

                    # Visualizar atributos para cómics
                    st.subheader('Para Cómics:')
                    visualizar_atributos_bar(ganadores_comics, perdedores_comics, atributos, 'Cómics')

                    # Visualizar atributos para películas
                    st.subheader('Para Películas:')
                    visualizar_atributos_bar(ganadores_peliculas, perdedores_peliculas, atributos, 'Películas')

                elif hipotesis_analisis == 'Distribucion comics vs peliculas':
                    st.write('Distribución de resultados en cómics vs películas')

                    # Crear un subset de los datos con los resultados de enfrentamientos en cómics y películas
                    resultados = df_pred_battles[['Resultado_Comics', 'Resultado_Peliculas']]

                    # Visualizar las distribuciones de los resultados
                    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                    sns.countplot(data=df_pred_battles, x='Resultado_Comics', palette='viridis', ax=axes[0])
                    axes[0].set_title('Distribución de Resultados en Cómics')
                    axes[0].set_xlabel('Resultado')
                    axes[0].set_ylabel('Frecuencia')

                    sns.countplot(data=df_pred_battles, x='Resultado_Peliculas', palette='viridis', ax=axes[1])
                    axes[1].set_title('Distribución de Resultados en Películas')
                    axes[1].set_xlabel('Resultado')
                    axes[1].set_ylabel('Frecuencia')

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Análisis Comparativo
                    # Crear una tabla de contingencia
                    tabla_contingencia = pd.crosstab(df_pred_battles['Resultado_Comics'], df_pred_battles['Resultado_Peliculas'])

                    # Prueba de chi-cuadrado
                    chi2, p, dof, expected = chi2_contingency(tabla_contingencia)

                    st.write("Tabla de contingencia entre resultados en cómics y películas:")
                    st.write(tabla_contingencia)
                    st.write(f"Chi-cuadrado: {chi2}")
                    st.write(f"P-valor: {p}")
                    st.write(f"Grados de libertad: {dof}")
                    st.write("Frecuencias esperadas:")
                    st.write(expected)

                elif hipotesis_analisis == 'Fuerza y Poder relacionados con la victoria':
                    st.write('Fuerza y Poder relacionados con la victoria')

                    # Datos de fuerza y poder para ganadores y perdedores
                    strength_winner_1 = df_comics_pelis[df_comics_pelis['Resultado_Combinado'] == 1]['Strength_1']
                    strength_loser_2 = df_comics_pelis[df_comics_pelis['Resultado_Combinado'] == 0]['Strength_2']
                    power_winner_1 = df_comics_pelis[df_comics_pelis['Resultado_Combinado'] == 1]['Power_1']
                    power_loser_2 = df_comics_pelis[df_comics_pelis['Resultado_Combinado'] == 0]['Power_2']

                    # Test t de Student para Strength
                    t_stat_strength_1, p_val_strength_1 = stats.ttest_ind(strength_winner_1, strength_loser_2)
                    st.write(f"Test t para Strength (Heroe 1 vs Heroe 2): t-stat = {t_stat_strength_1}, p-value = {p_val_strength_1}")

                    # Test t de Student para Power
                    t_stat_power_1, p_val_power_1 = stats.ttest_ind(power_winner_1, power_loser_2)
                    st.write(f"Test t para Power (Heroe 1 vs Heroe 2): t-stat = {t_stat_power_1}, p-value = {p_val_power_1}")

                elif hipotesis_analisis == 'Similitud heroes del mismo universo':
                    st.write('Similitud heroes del mismo universo')

                    # Crear una columna que indique si ambos héroes en un enfrentamiento pertenecen al mismo universo
                    df_comics_pelis['Mismo_Universo'] = df_comics_pelis['Universe_1'] == df_comics_pelis['Universe_2']

                    # Crear variables binarias para indicar si un héroe ganó en cómics y películas
                    df_comics_pelis['Ganador_Comics'] = (df_comics_pelis['Resultado_Comics'] == 1).astype(int)
                    df_comics_pelis['Ganador_Peliculas'] = (df_comics_pelis['Resultado_Peliculas'] == 1).astype(int)

                    # Visualizar las distribuciones de los patrones de victorias y derrotas
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                    sns.countplot(data=df_comics_pelis[df_comics_pelis['Mismo_Universo']], x='Ganador_Comics', palette='viridis', ax=axes[0, 0])
                    axes[0, 0].set_title('Distribución de Resultados en Cómics (Mismo Universo)')
                    axes[0, 0].set_xlabel('Ganador en Cómics')
                    axes[0, 0].set_ylabel('Frecuencia')

                    sns.countplot(data=df_comics_pelis[~df_comics_pelis['Mismo_Universo']], x='Ganador_Comics', palette='viridis', ax=axes[0, 1])
                    axes[0, 1].set_title('Distribución de Resultados en Cómics (Diferente Universo)')
                    axes[0, 1].set_xlabel('Ganador en Cómics')
                    axes[0, 1].set_ylabel('Frecuencia')

                    sns.countplot(data=df_comics_pelis[df_comics_pelis['Mismo_Universo']], x='Ganador_Peliculas', palette='viridis', ax=axes[1, 0])
                    axes[1, 0].set_title('Distribución de Resultados en Películas (Mismo Universo)')
                    axes[1, 0].set_xlabel('Ganador en Películas')
                    axes[1, 0].set_ylabel('Frecuencia')

                    sns.countplot(data=df_comics_pelis[~df_comics_pelis['Mismo_Universo']], x='Ganador_Peliculas', palette='viridis', ax=axes[1, 1])
                    axes[1, 1].set_title('Distribución de Resultados en Películas (Diferente Universo)')
                    axes[1, 1].set_xlabel('Ganador en Películas')
                    axes[1, 1].set_ylabel('Frecuencia')

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Análisis Comparativo
                    # Crear tablas de contingencia
                    contingencia_comics = pd.crosstab(df_comics_pelis['Mismo_Universo'], df_comics_pelis['Ganador_Comics'])
                    contingencia_peliculas = pd.crosstab(df_comics_pelis['Mismo_Universo'], df_comics_pelis['Ganador_Peliculas'])

                    # Realizar el test chi-cuadrado
                    chi2_comics, p_comics, dof_comics, expected_comics = chi2_contingency(contingencia_comics)
                    chi2_peliculas, p_peliculas, dof_peliculas, expected_peliculas = chi2_contingency(contingencia_peliculas)

                    st.write(f"Chi-cuadrado para cómics: {chi2_comics}")
                    st.write(f"p-valor para cómics: {p_comics}")
                    st.write(f"Grados de libertad para cómics: {dof_comics}")
                    st.write("Tabla esperada para cómics:")
                    st.write(expected_comics)

                    st.write(f"Chi-cuadrado para películas: {chi2_peliculas}")
                    st.write(f"p-valor para películas: {p_peliculas}")
                    st.write(f"Grados de libertad para películas: {dof_peliculas}")
                    st.write("Tabla esperada para películas:")
                    st.write(expected_peliculas)

        # Pestaña Simulador
        with tabs[1]:
            st.title('Simulador de batallas de superhéroes')

            col1, col2 = st.columns([1, 3])
            col3, col4 = st.columns([1, 3])

            # Selección de héroes
            with col1:
                heroe1 = st.selectbox('Selecciona el Héroe 1', ['Aleatorio'] + list(df_pred_battles['Heroe 1'].unique()))
            with col3:
                heroe2 = st.selectbox('Selecciona el Héroe 2', ['Aleatorio'] + list(df_pred_battles['Heroe 2'].unique()))

            # Mostrar imagen del Héroe 1
            with col2:
                mostrar_imagen_heroe(heroe1, 'Heroe 1')

            # Mostrar imagen del Héroe 2
            with col4:
                mostrar_imagen_heroe(heroe2, 'Heroe 2')

            # Obtener atributos del Héroe 1
            st.header('Atributos de Héroe 1')
            if heroe1 == 'Aleatorio':
                intelligence_1 = st.slider('Inteligencia', 0, 100, 50, key='intelligence_1')
                strength_1 = st.slider('Fuerza', 0, 100, 50, key='strength_1')
                speed_1 = st.slider('Velocidad', 0, 100, 50, key='speed_1')
                durability_1 = st.slider('Durabilidad', 0, 100, 50, key='durability_1')
                power_1 = st.slider('Poder', 0, 100, 50, key='power_1')
                combat_1 = st.slider('Combate', 0, 100, 50, key='combat_1')
                heroe1_encoded = random.randint(0, 500)  # Asignar un valor aleatorio
            else:
                atributos_1 = obtener_atributos(heroe1, 'Heroe 1')
                intelligence_1 = st.slider('Inteligencia', 0, 100, atributos_1['intelligence'], key='intelligence_1', disabled=True)
                strength_1 = st.slider('Fuerza', 0, 100, atributos_1['strength'], key='strength_1', disabled=True)
                speed_1 = st.slider('Velocidad', 0, 100, atributos_1['speed'], key='speed_1', disabled=True)
                durability_1 = st.slider('Durabilidad', 0, 100, atributos_1['durability'], key='durability_1', disabled=True)
                power_1 = st.slider('Poder', 0, 100, atributos_1['power'], key='power_1', disabled=True)
                combat_1 = st.slider('Combate', 0, 100, atributos_1['combat'], key='combat_1', disabled=True)
                heroe1_encoded = df_pred_battles.loc[df_pred_battles['Heroe 1'] == heroe1, 'Heroe_1_encoded'].values[0]

            # Obtener atributos del Héroe 2
            st.header('Atributos de Héroe 2')
            if heroe2 == 'Aleatorio':
                intelligence_2 = st.slider('Inteligencia', 0, 100, 50, key='intelligence_2')
                strength_2 = st.slider('Fuerza', 0, 100, 50, key='strength_2')
                speed_2 = st.slider('Velocidad', 0, 100, 50, key='speed_2')
                durability_2 = st.slider('Durabilidad', 0, 100, 50, key='durability_2')
                power_2 = st.slider('Poder', 0, 100, 50, key='power_2')
                combat_2 = st.slider('Combate', 0, 100, 50, key='combat_2')
                heroe2_encoded = random.randint(0, 500)  # Asignar un valor aleatorio
            else:
                atributos_2 = obtener_atributos(heroe2, 'Heroe 2')
                intelligence_2 = st.slider('Inteligencia', 0, 100, atributos_2['intelligence'], key='intelligence_2', disabled=True)
                strength_2 = st.slider('Fuerza', 0, 100, atributos_2['strength'], key='strength_2', disabled=True)
                speed_2 = st.slider('Velocidad', 0, 100, atributos_2['speed'], key='speed_2', disabled=True)
                durability_2 = st.slider('Durabilidad', 0, 100, atributos_2['durability'], key='durability_2', disabled=True)
                power_2 = st.slider('Poder', 0, 100, atributos_2['power'], key='power_2', disabled=True)
                combat_2 = st.slider('Combate', 0, 100, atributos_2['combat'], key='combat_2', disabled=True)
                heroe2_encoded = df_pred_battles.loc[df_pred_battles['Heroe 2'] == heroe2, 'Heroe_2_encoded'].values[0]

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
                ganador = simular_batalla(nuevo_combate, model)
                if ganador == "Heroe 1":
                    heroe_ganador = heroe1
                else:
                    heroe_ganador = heroe2

                st.write(f'El ganador es: {heroe_ganador}')
                mostrar_imagen_heroe(heroe_ganador, ganador)
