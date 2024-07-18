# SIMULADOR DE BATALLAS 

Bienvendio a **SUPERHERO BATTLE ARENA: THE ULTIMATE SHOWDOWN**

Esta innovadora aplicación te permite sumergirte en el fascinante mundo de los superhéroes, combinando el poder del análisis de datos y el machine learning. Con esta herramienta, podrás explorar detalladamente las características de tus héroes favoritos de Marvel y DC, simular enfrentamientos épicos y descubrir quién sería el vencedor en el combate definitivo. 
Nuestra aplicación utiliza técnicas avanzadas de Machine Learning y un exhaustivo Análisis Exploratorio de Datos (EDA) para brindarte una experiencia única y emocionante. 
Prepárate para experimentar la ciencia detrás de cada batalla y disfruta de la adrenalina de cada enfrentamiento en la Superhero Battle Arena.

## Resumen de datos 

La aplicacion utiliza un conjunto de datos detallados que incluye caracteristica clave de los superheroes. A continuacion, se presenta un pequeño resumen de los datos disponibles:

**Nombre**: Nombre del superheroe. 
**Strength**: Valor que representa la fuerza del superheroe. 
**Speed**: Valor que representa la velocidad del superhéroe.
**Durability**: Valor que representa la armadura del superhéroe.
**Power**: Valor que representa el poder general del superhéroe.
**Combat**: Valor que representa las habilidades de combate del superhéroe.

A continuacion, se presenta una muestra de los datos:

import pandas as pd

# Cargar el archivo CSV
df = './data/Data_final_MCU_vs_DC.csv'
data = pd.read_csv(file_path)

# Mostrar las primeras filas del dataframe
data_head = data.head()

# Generar la tabla en formato Markdown
markdown_table = data_head.to_markdown(index=False)
markdown_table
