import numpy as np
import pandas as pd
import cloudpickle

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Automcompletar rápido
#%config IPCompleter.greedy=True

from janitor import clean_names

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import seaborn as sns
from sklearn.metrics import confusion_matrix


############ CON ESTO FUNCIONA LA APP CON 4 VBLES ##########################################
# def ejecutar_modelo(df):
   
#     nombre_pipe_ejecucion = 'pipe_ejecucion.pickle'

#     with open(nombre_pipe_ejecucion, mode='rb') as file:
#         pipe_ejecucion = cloudpickle.load(file)

#     scoring = pipe_ejecucion.predict_proba(df)[:,1]

#     return 'El porcentaje de tener fallo cardiaco es: {}%'.format(round(scoring[0]*100,2))
############################################################################################

def ejecutar_modelo(df):
   
    nombre_pipe_ejecucion = 'pipe_ejecucion.pickle'

    with open(nombre_pipe_ejecucion, mode='rb') as file:
        pipe_ejecucion = cloudpickle.load(file)

    scoring = pipe_ejecucion.predict_proba(df)[:,1]
    return scoring[0]




def max_roi_5(real,scoring, salida = 'grafico'):

    #DEFINIMOS LA FUNCION DEL VALOR ESPERADO
    def valor_esperado(matriz_conf):
        TN, FP, FN, TP = conf.ravel()
        VE = (TN * ITN_usu) + (FP * IFP_usu) + (FN * IFN_usu) + (TP * ITP_usu)
        return(VE)
    
    #CREAMOS UNA LISTA PARA EL VALOR ESPERADO
    ve_list = []
    
    #ITERAMOS CADA PUNTO DE CORTE Y RECOGEMOS SU VE
    for umbral in np.arange(0,1,0.01):
        predicho = np.where(scoring > umbral,1,0) 
        conf = confusion_matrix(real,predicho)
        ve_temp = valor_esperado(conf)
        ve_list.append(tuple([umbral,ve_temp]))
        
    #DEVUELVE EL RESULTADO COMO GRAFICO O COMO EL UMBRAL ÓPTIMO
    df_temp = pd.DataFrame(ve_list, columns = ['umbral', 'valor_esperado'])
    if salida == 'grafico':
#         solo_ve_positivo = df_temp[df_temp.valor_esperado > 0]
#         plt.figure(figsize = (12,6))
#         sns.lineplot(data = solo_ve_positivo, x = 'umbral', y = 'valor_esperado')
#         plt.xticks(solo_ve_positivo.umbral, fontsize = 14)
#         plt.yticks(solo_ve_positivo.valor_esperado, fontsize = 12);
        
        plt.figure(figsize = (12,6))
        sns.lineplot(data = df_temp, x = 'umbral', y = 'valor_esperado')
        plt.xticks(df_temp.umbral, fontsize = 14)
        plt.yticks(df_temp.valor_esperado, fontsize = 12);
    else:    
        return(df_temp.iloc[df_temp.valor_esperado.idxmax(),0])
        #return(df_temp.iloc[df_temp.valor_esperado.idxmin(),0])












