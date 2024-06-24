import bootcampviztools as bt
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import toolbox_ML as tb
import warnings

warnings.filterwarnings('ignore')

from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


def paramns_check(df:pd.DataFrame, target_col:str, columns:list, pvalue:float) -> bool:
    
    """
    Esta es una funcion de comprobacion para los parametros.

    Comprobamos que:

    .- el parametro df es un dataframe de pandas
    .- el target seleccionado es categorico, definido por un str que referencia clases, en caso de ser numerico corresponderia mapearlo a str
    .- que las columnas proporcionadas son numericas 
    .- que el pvalue es numerico y esta entre 0 y 1

    La función devuelve un booleano que certifica si los parametros introducidos son adecuados.
    """
    
    try:
        if type(df) != pd.core.frame.DataFrame:
            return False
        if df[target_col].dtype != 'object':
            return False
        for col in columns:
            pd.to_numeric(df[col])
        if (float(pvalue) > 1) or (float(pvalue) < 0):
            return False
    except:
        return False
    
    return True

def eval_model(target, predictions, problem_type, metrics):
        
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def eval_model(target, predictions, problem_type, metrics):
        
        """
    Evalúa un modelo de regresión o clasificación en base a las métricas especificadas.

    Argumentos:
    - target (array-like): Valores verdaderos de los datos objetivo.
    - predictions (array-like): Valores predichos por el modelo.
    - problem_type (str): Tipo de problema, 'regression' para regresión o 'classification' para clasificación.
    - metrics (list of str): Lista de métricas a evaluar. Las métricas posibles dependen del tipo de problema.

    Retorna:
    - tuple: Tupla con los resultados de las métricas solicitadas, en el orden en que aparecen en la lista de entrada.
    """
        results = []

        if problem_type == 'regression':
            if not all(metric in ['RMSE', 'MAE', 'MAPE', 'GRAPH'] for metric in metrics):
                raise ValueError('Las metricas para regresion deben ser "RMSE", "MAE", "MAPE", "GRAPH".')
            
            for metric in metrics:
                if metric == 'RMSE':
                    rmse = np.sqrt(mean_squared_error(target, predictions))
                    print(f'RMSE: {rmse}')
                    results.append(rmse)
                elif metric == 'MAE':
                    mae = mean_absolute_error(target, predictions)
                    print(f'MAE: {mae}')
                    results.append(mae)
                elif metric == 'MAPE':
                    try:
                        mape = np.mean(np.abs((target - predictions / target))) * 100
                    except ZeroDivisionError:
                         raise ValueError('No se puede calcuar el MAPE porque el target contiene valores 0')
                    print(f'MAPE: {mape}')
                    results.append(mape)
                elif metric == 'GRAPH':
                    plt.figure(figsize = (12, 8))
                    plt.scatter(target, predictions, alpha = 0.3)
                    plt.xlabel('Target')
                    plt.ylabel('Predictions')
                    plt.title('Target Vs Predictions')
                    plt.grid(True)
                    plt.show()
        
        elif problem_type == 'classification':
            print(type(metrics))
            if not all(metric.startswith(('ACCURACY', 'PRECISION', 'RECALL', 'CLASS REPORT', 'MATRIX')) for metric in metrics):
                raise ValueError('Las metricas para regresion deben ser "ACCURACY", "PRECISION", "RECALL", "CLASS_REPORT", "MATRIX", "MATRIX_RECALL", "MATRIX_PRED" o "PRECISION_X", "RECALL_X".')
            
            for metric in metrics:
                if metric == 'ACCURACY':
                    accuracy = accuracy_score(target, predictions)
                    print(f'Accuracy: {accuracy}')
                    results.append(accuracy)
                elif metric == 'PRECISION':
                    precision = precision_score(target, predictions, average = 'macro')
                    print(f'Precision: {precision}')
                    results.append(precision)
                elif metric == 'RECALL':
                    recall = recall_score(target, predictions, average = 'macro')
                    print(f'Recall: {recall}')
                    results.append(recall)
                elif metric == 'CLASS_REPORT':
                    report = classification_report(target, predictions)
                    print('Classification Report')
                    print(report)
                elif metric == 'MATRIX':
                    con_matrix = confusion_matrix(target, predictions)
                    print('Confusion Matrix')
                    print(con_matrix)
                    disp = ConfusionMatrixDisplay(confusion_matrix = con_matrix)
                    disp.plot()
                    plt.show()
                elif metric == 'MATRIX_RECALL':
                    mat_rec = confusion_matrix(target, predictions, normalize = True)
                    print('Confusion Matrix Normalize Recall')
                    print(mat_rec)
                    disp = ConfusionMatrixDisplay(confusion_matrix = mat_rec)
                    disp.plot()
                    plt.show()
                elif metric == 'MATRIX_PRED':
                    mat_pred = confusion_matrix(target, predictions, normalize = 'pred')
                    print(f'Confusion Matrix Normalize Predictions')
                    print(mat_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix = mat_pred)
                    disp.plot()
                    plt.show()
                elif metric.startswith('PRECISION_'):
                    label = metric.split('_')[1]
                    try:
                        precision = precision_score(target, predictions, labels = [label], average = 'micro')
                        print(f'Precision for class "{label}: {precision}')
                        results.append(precision)
                    except ValueError:
                        raise ValueError(f'La clase "{label}" no esta presente en el target.')
                elif metric.startswith('RECALL_'):
                    label = metric.split(_)[1]
                    try:
                        recall = recall_score(target, predictions, labels = [label], average = 'micro')
                        print(f'Recall for class "{label}": {recall}')
                        results.append(recall)
                    except ValueError:
                        raise ValueError(f'La clase "{label}" no esta presente en el target.')
        else:
            raise ValueError('El tipo de problema debe ser "regression" o "classification".')
        
        return tuple(results)

def get_features_num_classification(df, target_col, pvalue=0.05):

    """
    Identifica columnas numéricas en un DataFrame que tienen un resultado significativo
    en la prueba ANOVA con respecto a una columna objetivo categórica.

    Parámetros:
    df (pd.DataFrame): El DataFrame de entrada que contiene los datos.
    target_col (str): El nombre de la columna objetivo en el DataFrame. Esta debe ser 
                      una columna categórica con baja cardinalidad (10 o menos valores únicos).
    pvalue (float): El nivel de significancia para la prueba ANOVA. El valor predeterminado es 0.05.

    Retorna:
    list: Una lista de nombres de columnas numéricas que tienen una relación significativa con 
          la columna objetivo según la prueba ANOVA.
          Retorna None si alguna de las comprobaciones de los argumentos de entrada falla, 
          e imprime un mensaje indicando la razón.
    """
    
    # Comprobación de que el DataFrame no está vacío
    if df.empty:
        print("El DataFrame está vacío.")
        return None
    
    # Comprobación de que target_col está en el DataFrame
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no se encuentra en el DataFrame.")
        return None
    
    # Comprobación de que target_col es categórica con baja cardinalidad
    if not isinstance(df[target_col].dtype, pd.CategoricalDtype) and not pd.api.types.is_object_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es categórica.")
        return None
    
    if df[target_col].nunique() > 10:
        print(f"La columna '{target_col}' tiene demasiadas categorías (más de 10).")
        return None
    
    # Comprobación de que pvalue es un float y está en el rango correcto
    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print("El valor de 'pvalue' debe ser un float entre 0 y 1.")
        return None
    
    # Filtrar las columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Lista para almacenar las columnas que cumplen con el criterio
    significant_columns = []
    
    for col in numeric_cols:
        groups = [df[col][df[target_col] == category] for category in df[target_col].unique()]
        f_stat, p_val = f_oneway(*groups)
        if p_val <= pvalue:
            significant_columns.append(col)
    
    return significant_columns

def plot_features_num_classification(df:pd.DataFrame, target_col:str= '', columns:list= [], pvalue:float= 0.05) -> list:
    # version con generador de indices
    """
    Parámetros:
    .- df: un dataframe de pandas
    .- target_col: el nombre de la variable target (debe ser categorica objeto/str, si contiene numeros, procede mapearla)
    .- columns: el nombre de las variables numericas del df, adjuntas en una lista (vacia por defecto)
    .- pvalue: el valor con que queremos comprobar la significancia estadistica, 0.05 por defecto

    Esta funcion cumple tras objetivos: a saber:

    1.- retorna una lista con los nombres de las features numericas que superan un test anova de significancia estadistica superior al establecido en pvalue
    2.- printa una relacion de graficas comparativas de correlacion target-variables numericas para su estudio y miniEDA
    3.- printa una relacion de graficas comparativas de colinealidad entre las distinta variables numericas para su estudio y miniEDA

    Explicamos la funcion mas en detalle a continuacion.
    """

    paramns_ok = paramns_check(df, target_col, columns, pvalue) # comprobamos que los parametros son adecuados, si no lo son retornamos None y printamos que no lo son
    if not paramns_ok:
        print('Los parametros introduciodos son incorrectos.')
        return None

    if not columns: # si no adjuntamos lista de var numericas, cogemos todas las numericas del df
        columns = df.describe().columns.tolist()

    col_anova = [] # creamos lista vacia donde almacenaremos los nombres de var numericas que cumplen el test anova

    # a continuacion realizamos el test anova
    grps = df[target_col].unique().tolist() # almacenamo los diferentes valores posibles del target en una lista
    for feature in columns: # iteramos las var numricas
        prov_list = [] # lista provisional donde almacenaremos las series de realcion de cada var numrica con los diferentes valores del target
        
        for grp in grps:
            prov_list.append(df[df[target_col] == grp][feature]) # agregamos a la lista las series que comentabamos antes
        
        f_st, p_va = f_oneway(*prov_list) # realizamos el test anova sobre la var numerica de turno (en iteracion actual) en relacion con cada valor del target y comprobamos su pvalue en funcion de su varianza
        if p_va <= pvalue: # si hay significancia estadistica recahazamos H0(medias similares) y adjuntamos el nombre de la feature a col_anova 
            col_anova.append(feature) 
    
    # empezamos con las graficas
    col_anova.insert(0, target_col) # adjuntamos el target a col_anova porque lo necesitaremos para comparar y graficar

    # creamos una primera serie de graficas relacion target(categorica) con las features numericas
    # utilizaremos subplots para reflejar cada grafica individualmente. Estos subplots son referenciados mediante arrays, importante

    q_lineas = math.ceil((len(col_anova)-1)/5) # calculamos la cantidad de lineas que compondra en la figura grafica / array (cada linea comprendera 5 subplots / columnas)

    # vamos a jugar con generadores, uno simple en realidad, no lo hemos visto en temario pero para este caso resulta de mucha utilidad
    # para movernos por los subplots de la figura grafica de turno deberemos iterar las columnas segun grafiquemos diferentes relaciones target-features
    # este generador genera los indices para el subplot
    def gen_indice():
        
        while True:
            for linea in range(100):
                for columna in range(5):
                    yield linea, columna

    contador_indice = gen_indice() # instanciamos el generador


    fig, axs = plt.subplots(q_lineas, 5, figsize=(20, 4*q_lineas)) # generamos la figura grafica con la cantidad de lineas y 5 columnas, tamño acorde a la q de lineas
    fig.suptitle('Correlación target categorico VS features numéricas con significancia estadistica > 1-pvalue')
    plt.subplots_adjust(top=0.9)

    columna = 0 # comenzamos en la linea 0, primera
    indice = next(contador_indice) # primer indice [0, 0]
    # comenzamos a iterar las features que tenemos que graficas
    for feature_index in range(1, len(col_anova)): # rango 1 hata final porque la primera es el target y no queremos graficar target-target
    
        try: # presumimos que la grafica dispondra de mas de 1 linea y graficaremos en consecuencia... 
            for i in df[col_anova[0]].unique():     
                sns.histplot(df[df[col_anova[0]] == i][col_anova[feature_index]], kde= True, ax= axs[indice], label= i)
            axs[indice].legend()
            indice = next(contador_indice) # siguiente indice
        except IndexError: # ...si la figura solo dispone de 1 linea la graficacion dara error y graficamos en consecuencia
            for i in df[col_anova[0]].unique():     
                sns.histplot(df[df[col_anova[0]] == i][col_anova[feature_index]], kde= True, ax= axs[columna], label= i)
            axs[columna].legend()
            columna += 1 # siguiente columna
    plt.show() # mostramos la figura grafica

    # graficamos la colinealidad
    sns.pairplot(df[col_anova], hue= target_col) # pairplot para todas las features numericas que han superado la significancia estadistica
    plt.suptitle('Colinealidad features numéricas con significancia estadistica > 1-pvalue')
    plt.subplots_adjust(top=0.9) 
    plt.show() # mostramos grafica
    col_anova.remove(target_col) # quitamos el target de la lista de features que han superado el test (ya ha sido util para graficar)
    
    return col_anova # devolvemos los nombres de las features que han superado la significancia estadistica

def plot_features_cat_classification(df, target_col="", columns=[], mi_threshold=0.0, normalize=False):
    
    """
    Pinta las distribuciones de las columnas categoricas que pasan un threshold de informacion mutua con respecto a una columna objetivo haciendo uso de la funcionget_features_cat_classification
    
    Parámetros: 
    - df->dataframe objetivo 
    - target_col->columna(s) objetivo, pueden ser varias
    - mi_threshold->limite usado para la comprobacion de informacion mutua de las columnas
    - normalize->booleano que indica si se ha de normalizar o no a la hora de comprobar la informacion mutua
    
    Rertorna:
    - Plots de las variables que han pasado el limite de informacion mutua, representando la relacion entre esa columna y la columna objetivo
    """
    if not isinstance(df, pd.DataFrame):
        print("El dataframe proporcionado en realidad no es un dataframe")
        return None
    
    if target_col == "":
        print("Especifica una columna")
        return None
    
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no esta en el datarame")
        return None

    if not isinstance(df[target_col].dtype, pd.CategoricalDtype):
        df[target_col] = df[target_col].astype('category')
    
    if not columns:
        columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
        if target_col in columns:
            columns.remove(target_col)
    
    if not all(col in df.columns for col in columns):
        print("Comprueba que todas las columnas espeficadas esten en el dataframe")
        return None
    
    selected_columns = get_features_cat_classification(df, target_col, normalize, mi_threshold)
    
    if not selected_columns:
        print("Ninguna columna cumple con la condicion de la informacion mutua")
        return None
    
    for col in selected_columns:
        plt.figure(figsize=(10, 6))
        df.groupby([col, target_col]).size().unstack().plot(kind='bar', stacked=True)
        plt.title(f'Distribucion de {target_col} con {col}')
        plt.xlabel(col)
        plt.ylabel('Contador')
        plt.legend(title=target_col)
        plt.show()



def get_features_cat_classification(dataframe, target_col, normalize=False, mi_threshold=0.0):
    # Validar el DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        print("Error: El primer argumento debe ser un DataFrame de pandas.")
        return None
    
    # Validar el nombre de la columna objetivo
    if target_col not in dataframe.columns:
        print(f"Error: '{target_col}' no es una columna del DataFrame.")
        return None
    
    # Verificar que la columna objetivo es categórica o numérica discreta de baja cardinalidad
    if not (isinstance(dataframe[target_col].dtype, pd.CategoricalDtype) or 
            (dataframe[target_col].dtype in ['int64', 'float64', 'object'] and dataframe[target_col].nunique() < 20)):
        print(f"Error: La columna '{target_col}' debe ser categórica o numérica discreta con baja cardinalidad.")
        return None
    
    # Validar que normalize es un booleano
    if not isinstance(normalize, bool):
        print("Error: El argumento 'normalize' debe ser un booleano.")
        return None
    
    # Validar que mi_threshold es un número
    if not isinstance(mi_threshold, (int, float)):
        print("Error: El argumento 'mi_threshold' debe ser un número.")
        return None
    
    # Validar el rango de mi_threshold si normalize es True
    if normalize and (mi_threshold < 0 or mi_threshold > 1):
        print("Error: 'mi_threshold' debe ser un valor flotante entre 0 y 1 cuando 'normalize' es True.")
        return None
    
    # Seleccionar características categóricas y numéricas discretas
    cat_features = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_features = [col for col in cat_features if col != target_col]
    
    # Codificar características categóricas
    X = pd.get_dummies(dataframe[cat_features])
    y = dataframe[target_col].astype('category').cat.codes  # Codificar la columna objetivo como categorías numéricas

    # Cálculo de la información mutua
    mi = mutual_info_classif(X, y, discrete_features=True)
    
    # Normalización de la información mutua si se requiere
    if normalize:
        total_mi = sum(mi)
        if total_mi == 0:
            print("Error: La suma de los valores de información mutua es 0, no se puede normalizar.")
            return None
        mi = mi / total_mi
    
    # Seleccionar características basadas en el umbral
    selected_features = [cat_features[i] for i in range(len(cat_features)) if mi[i] >= mi_threshold]
    
    return selected_features       