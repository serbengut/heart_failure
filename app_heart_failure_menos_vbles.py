import pandas as pd

import streamlit as st
#from streamlit_echarts import st_echarts
from codigo_ejecucion_heart_failure_menos_variables import *
import numpy as np
import cloudpickle
import pickle

from janitor import clean_names

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import HistGradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

st.set_page_config(
     page_title = 'Heart Failure',
     #page_icon = 'DS4B_Logo_Blanco_Vertical_FB.png',
     #layout = 'wide')
)

st.title('Heart Failure')


### vble 1:
#age = st.number_input(label='Edad:',step=1.,format="%.0f")

### vble 2:
chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY','TA'])

### vble 3:
#cholesterol = st.slider('Cholesterol level', 70, 800)

### vble 4:
#exercise_angina = st.selectbox('Exercise Angina', ['Y', 'N'])

### vble 5:
#fasting_bs = st.selectbox('Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]', [1, 0])

### vble 6:
#max_hr = st.slider('Max heart rate', 80, 300)

### vble 7:
oldpeak = st.number_input(label='OldPeak', min_value=-2.0, max_value=6.0,step=0.1,format="%.2f")
#oldpeak=float(oldpeak)

### vble 8:
#resting_bp = st.slider('Resting heart rate', 80, 200)

### vble 9:
#resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST','LVH'])

### vble 10:
sex = st.selectbox('Sex', ['M', 'F'])

### vble 11:
st_slope = st.selectbox('Slope of the peak exercise:', ['Up', 'Flat','Down']) 


registro = pd.DataFrame({'chest_pain_type':chest_pain_type,                                              
                         'sex':sex,
                         'oldpeak':oldpeak,
                         'st_slope':st_slope
                         }
                        ,index=[0])

registro


coste_tto_preventivo = st.slider('Coste tto. preventivo', 500, 3000)
coste_tto = st.slider('Coste tratamiento paciente enfermo a largo plazo: ', 3000, 100000)

ITN_usu = 0
IFP_usu = -coste_tto_preventivo
IFN_usu = -coste_tto + coste_tto_preventivo
ITP_usu = +coste_tto - coste_tto_preventivo

############ CON ESTO FUNCIONA LA APP CON 4 VBLES ##########################################
# if st.sidebar.button('CALCULAR POSIBILIDAD DE FALLO CARDIACO'):

#     fallo = ejecutar_modelo(registro)

#     fallo

# else:
#     st.write('DEFINE LOS PARÁMETROS Y HAZ CLICK EN CALCULAR POSIBILIDAD DE FALLO CARDIACO')
############################################################################################

def carga_x_y():
    pred = np.loadtxt('pred_final.txt')
    val_y_final = np.loadtxt('val_y_final_.txt')
    return pred, val_y_final

########################################################################################
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
    

##### boton calcular prob fallo cardiaco: ##########################
if st.sidebar.button('CALCULAR POSIBILIDAD DE FALLO CARDIACO y MAX ROI'):
    fallo = ejecutar_modelo(registro)
    st.write(f'Probabilidad de fallo cardicaco: {round(100*fallo,2)}%')

    pred, val_y_final = carga_x_y()

    umbral_usu = max_roi_5(val_y_final, pred,salida = 'automatico')

    st.write(f'El umbral para decidir si aplicar o no tratamiento preventivo es: {100*(umbral_usu)}%')

    st.write(f'Como la probabilidad de que el paciente tenga fallo cardiaco es: {round(100*fallo,2)}%')

    st.write('Desde el punto de vista del coste del tto. preventivo versus coste del tto. normal:')
    if fallo > umbral_usu:
        st.subheader('PACIENTE ELEGIBLE PARA HACER TRATAMIENTO PREVENTIVO')
    else:
        st.subheader('PACIENTE NO ELEGIBLE PARA HACER TRATAMIENTO PREVENTIVO')



else:
    st.write('DEFINE LOS PARÁMETROS Y HAZ CLICK EN CALCULAR POSIBILIDAD DE FALLO CARDIACO')
