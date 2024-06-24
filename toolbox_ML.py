import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency

def describe_df(df):
    """
    Genera un resumen estadístico de un dataframe proporcionando información sobre el tipo de datos,
    porcentaje de valores nulos, valores únicos y cardinalidad de cada columna, pero con las filas
    y columnas completamente intercambiadas respecto a la versión inicial.

    Argumentos:
    - df (DataFrame de pandas): El dataframe a describir.

    Retorna:
    - DataFrame: Un nuevo dataframe con las estadísticas de cada columna del dataframe original,
      con las estadísticas como columnas y las características de los datos como filas.
    """

    # Preparar el dataframe de resumen con filas y columnas intercambiadas
    summary = pd.DataFrame({
        'Data type': df.dtypes,
        'Percent missing (%)': df.isna().mean() * 100,
        'Unique values': df.nunique(),
        'Cardinality percent (%)': (df.nunique() / len(df)) * 100
    })

    return summary.transpose()  # Transponer el resultado 

def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Esta función analiza las columnas de un DataFrame para sugerir el tipo de variable que representan.
    Utiliza la cardinalidad y el porcentaje de cardinalidad de cada columna para determinar si se trata
    de una variable binaria, categórica, numérica continua o numérica discreta.
    
    Argumentos:
    - df (DataFrame de pandas): El DataFrame que contiene las variables a analizar.
    - umbral_categoria (int): Umbral que define el límite máximo de cardinalidad para considerar
      una variable como categórica. Si la cardinalidad de una columna es menor que este umbral, se
      sugiere que la variable es categórica.
    - umbral_continua (float): Umbral que define el porcentaje mínimo de cardinalidad para considerar
      una variable como numérica continua. Si la cardinalidad de una columna es mayor o igual que
      `umbral_categoria` y el porcentaje de cardinalidad es mayor o igual que este umbral, se sugiere
      que la variable es numérica continua.
      
    Retorna:
    - DataFrame: Un DataFrame que contiene dos columnas: "nombre_variable" y "tipo_sugerido". Cada
      fila del DataFrame representa una columna del DataFrame de entrada, con el nombre de la columna
      y el tipo sugerido de variable.
    """

    # Inicializar una lista para almacenar los resultados
    resultados = []
    
    # Iterar sobre cada columna del dataframe
    for columna in df.columns:
        # Calcular la cardinalidad de la columna
        cardinalidad = df[columna].nunique()
        
        # Calcular el porcentaje de cardinalidad
        porcentaje_cardinalidad = cardinalidad / len(df)
        
        # Determinar el tipo de variable
        if cardinalidad == 2:
            tipo_sugerido = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = "Categórica"
        else:
            if porcentaje_cardinalidad >= umbral_continua:
                tipo_sugerido = "Numerica Continua"
            else:
                tipo_sugerido = "Numerica Discreta"
        
        # Agregar el resultado a la lista
        resultados.append({'nombre_variable': columna, 'tipo_sugerido': tipo_sugerido})
    
    # Convertir la lista de resultados en un DataFrame y devolverlo
    return pd.DataFrame(resultados)

def get_features_num_regresion(df, target_col, umbral_corr, pvalue= None):
    """
    Esta función devuelve las features para la creacion de un modelo de machine learning.

    Estas features deben ser variables numericas y disponer de una correlacón y significacion estadistica significativa
    con el target, definidos previamente por el usuario. La significacion estadistica es nula por defecto.

    Argumentos:
    - df (DataFrame de pandas): un dataframe pandas sobre el que realizar el estudio.
    - target_col (str): la columna seleccionada como target para nuestro modelo.
    - umbral_corr (float): la correlacion minima exigida a una variable con el target para ser designado como feature. 
      Debe estar comprendido entre 0 y 1.
    - pvalue (float o None): la significacion estadistica Pearson maxima exigida a una variable para ser designada como feature 
      (generalmente 0.005). Por defecto, es None

    Retorna:
    - Lista con las columnas designadas como features para el modelo. Tipo lista compuesto por cadenas de texto.
    """

    cardinalidad = df[target_col].nunique() / len(df[target_col])

    if (umbral_corr < 0) or (umbral_corr > 1):

        print('Variable umbral_corr incorrecto.')
        return None

    elif df[target_col].dtype not in ['int8', 'int16', 'int32','int64', 'float16', 'float32', 'float64']:

        print('La columna seleccionada como target debe ser numerica.')
        return None
    
    elif cardinalidad < 0: # este no se si ponerlo

        print('Tu variable target tiene una cardinalidad muy baja para ser target.')
        return None
    
    lista_numericas = []
    for column in df.columns:
        
        if df[column].dtypes in ['int8', 'int16', 'int32','int64', 'float16', 'float32', 'float64']:
            lista_numericas.append(column)

    lista_numericas.remove(target_col)
    lista_features = []
    for columna in lista_numericas:

        no_nulos = df.dropna(subset= [target_col, columna])
        corr, pearson = pearsonr(no_nulos[target_col], no_nulos[columna])

        if pvalue != None:
            if (abs(corr) >= umbral_corr) and (pearson <= pvalue):
                lista_features.append(columna)
        else:
            if abs(corr) >= umbral_corr:
                lista_features.append(columna)
    
    return lista_features

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):

    """
    Esta función realiza una serie de comprobaciones de validez sobre los argumentos de entrada, como si el primer argumento es un DataFrame, si la columna objetivo está presente en el DataFrame y si las columnas especificadas para considerar son válidas. Luego, filtra las columnas numéricas basadas en su correlación con la columna objetivo y, opcionalmente, en el valor de p-value.

    Argumentos:
    - df (DataFrame de pandas): El DataFrame sobre el que se realizará el análisis.
    - target_col (str): La columna objetivo que se utilizará en el análisis de correlación.
    - columns (lista de str): La lista de columnas a considerar en el análisis de correlación.
    - umbral_corr (float): El umbral de correlación mínimo requerido para que una variable sea considerada relevante. Debe estar entre 0 y 1.
    - pvalue (float o None): El valor de p-value máximo aceptable para que una variable sea considerada relevante. Por defecto, es None.

    La función luego divide las columnas filtradas en grupos de hasta 4 y genera pairplots utilizando `sns.pairplot()`, mostrando las relaciones entre estas variables y la columna objetivo. Finalmente, devuelve una  lista de las columnas filtradas que cumplen los criterios de correlación y p-value. Si no hay variables que cumplan los criterios, imprime un mensaje de error y devuelve None.
    """

    # Comprobación de valores de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: El primer argumento debe ser un DataFrame.")
        return None
    
    if target_col not in df.columns:
        print("Error: 'target_col' debe ser una columna válida del DataFrame.")
        return None
    
    if not isinstance(columns, list):
        print("Error: 'columns' debe ser una lista de nombres de columnas.")
        return None
    
    for col in columns:
        if col not in df.columns:
            print(f"Error: '{col}' no es una columna válida del DataFrame.")
            return None
    
    if not isinstance(umbral_corr, (int, float)):
        print("Error: 'umbral_corr' debe ser un valor numérico.")
        return None
    
    if not isinstance(pvalue, (float, int, type(None))):
        print("Error: 'pvalue' debe ser un valor numérico o None.")
        return None
    
    if not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar en el rango [0, 1].")
        return None
    
    # Verificar que target_col sea una variable numérica continua del DataFrame
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: 'target_col' debe ser una variable numérica continua del DataFrame.")
        return None

    # Si la lista de columnas está vacía, seleccionar todas las variables numéricas
    if not columns:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Filtrar columnas según correlación y p-value si se proporcionan
    filtered_columns = []
    for col in columns:
        if col != target_col:
            correlation = pearsonr(df[target_col], df[col])[0]
            if abs(correlation) > umbral_corr:
                if pvalue is not None:
                    _, p_val = pearsonr(df[target_col], df[col])
                    if p_val < (1 - pvalue):
                        filtered_columns.append(col)
                else:
                    filtered_columns.append(col)
    
    if not filtered_columns:
        print("No hay variables que cumplan los criterios de correlación y p-value.")
        return None
    
    # Dividir las columnas filtradas en grupos de máximo 4 para pintar pairplots
    num_plots = (len(filtered_columns) // 3) + 1
    for i in range(num_plots):
        cols_to_plot = [target_col] + filtered_columns[i*3:(i+1)*3]
        sns.pairplot(df[cols_to_plot])
        plt.show()
    
    return filtered_columns

def get_features_cat_regression(df, target_col, p_value=0.05):
    """
    Identifica características categóricas relevantes para un modelo de regresión.

    Argumentos:
    - df (DataFrame de pandas): El DataFrame sobre el que se realizará el análisis.
    - target_col (str): La columna objetivo que se utilizará en el análisis.
    - p_value (float): El valor de p máximo aceptable para considerar una característica como relevante.
      Por defecto, es 0.05.

    Retorna:
    - Lista con las columnas categóricas consideradas relevantes para el modelo de regresión.
      Tipo lista compuesto por cadenas de texto.
    """

    if df.empty:
        print("El dataframe esta vacío")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("La columna que has puesto no es una columna numerica")
        return None
    if not isinstance(p_value, float) or 0 > p_value or 1 < p_value:
        print("El p_value no tiene un valor valido, recuerda que tiene que estar entre 0 y 1")
        return None
    if target_col not in df:
        print("La columna no esta en el Dataframe, cambiala por una valida")
        return None
    
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    relevant_columns = []
    
    for col in categorical_columns:
        grouped = df.groupby(col)[target_col].apply(list).to_dict()
        f_vals = []
        for key, value in grouped.items():
            f_vals.append(value)
        f_val, p_val = stats.f_oneway(*f_vals)
        if p_val <= p_value:
            relevant_columns.append(col)

    return relevant_columns


def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Realiza un análisis de las características categóricas en relación con una columna objetivo para un modelo de regresión.

    Argumentos:
    - df (DataFrame de pandas): El DataFrame que contiene los datos.
    - target_col (str): La columna objetivo para el análisis.
    - columns (list): Lista de columnas categóricas a considerar. Si está vacía, se considerarán todas las columnas categóricas del DataFrame.
    - pvalue (float): El nivel de significancia para determinar la relevancia estadística de las variables categóricas. Por defecto, es 0.05.
    - with_individual_plot (bool): Indica si se debe mostrar un histograma agrupado para cada variable categórica significativa. Por defecto, es False.

    Retorna:
    - Lista de las columnas categóricas que muestran significancia estadística con respecto a la columna objetivo.
    """

    # Verificar que dataframe sea un DataFrame de pandas
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El argumento 'dataframe' debe ser un DataFrame de pandas")

    # Verificar que target_col esté en el dataframe
    if target_col != "" and target_col not in df.columns:
        raise ValueError("La columna 'target_col' no existe en el DataFrame")

    # Verificar que las columnas en columns estén en el dataframe
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame")

    # Verificar que pvalue sea un valor válido
    if not isinstance(pvalue, (int, float)):
        raise ValueError("El argumento 'pvalue' debe ser un valor numérico")
    
    # Verificar que with_individual_plot sea un valor booleano
    if not isinstance(with_individual_plot, bool):
        raise ValueError("El argumento 'with_individual_plot' debe ser un valor booleano")

    # Si columns está vacío, seleccionar todas las variables categóricas
    if not columns:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    # Lista para almacenar las variables categóricas que cumplen con las condiciones
    significant_categorical_variables = []

    # Iterar sobre las columnas seleccionadas
    for col in columns:
        # Verificar si la columna es categórica
        if df[col].dtype == 'object':
            # Calcular el test de chi-cuadrado entre la columna categórica y target_col
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, p_val, _, _ = chi2_contingency(contingency_table)
            
            # Verificar si el p-value es menor que el umbral de significancia
            if p_val < pvalue:
                # Agregar la columna a la lista de variables categóricas significativas
                significant_categorical_variables.append(col)

                sns.histplot(data=df, x=col, hue=target_col, multiple="stack")
                plt.title(f"Histograma agrupado de {col} según {target_col}")
                plt.show()
            else:
                print(f"No se encontró significancia estadística para la variable categórica '{col}' con '{target_col}'")

    # Si no se encontró significancia estadística para ninguna variable categórica
    if not significant_categorical_variables:
        print("No se encontró significancia estadística para ninguna variable categórica")

    # Devolver las variables categóricas que cumplen con las condiciones
    return significant_categorical_variables