# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.tree
from sklearn import ensemble
from time import time
import pickle
import logging

# import xgboost as xgb

input_path = '/home/ubuntu/Data/'
output_path = '/home/ubuntu/Data/Result/'
log_path = '/home/ubuntu/Data/Logging/'
local_path = 'D:/Codes/Data/SantanderProductRecommendation/'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

feature_cols = ['ncodpers', 'ind_empleado', 'pais_residencia', 'sexo', 'age', 'ind_nuevo', 'antiguedad', 'indrel',
                'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext', 'canal_entrada', 'indfall', 'nomprov',
                'ind_actividad_cliente', 'renta', 'segmento', 'month']

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
               'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']


def process_data():
    df = pd.read_csv(input_path + 'trainLabelEncode.csv', usecols=['ncodpers']+target_cols)
    df.drop_duplicates(['ncodpers'], keep='last', inplace=True)
    dft = pd.read_csv(input_path + 'Result/sample_submission.csv')


if __name__ == '__main__':
    st = time()
    et = time()
    print('the program cost %d s' % (et-st))

