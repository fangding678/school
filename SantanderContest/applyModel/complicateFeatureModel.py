# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import ensemble
import xgboost as xgb
from time import time
import pickle
import logging

input_path = '/home/ubuntu/Data/'
output_path = '/home/ubuntu/Data/Result/'
log_path = '/home/ubuntu/Data/Logging/'
local_path = 'D:/Codes/Data/SantanderProductRecommendation/'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

feature_cols = ['ind_empleado', 'pais_residencia', 'sexo', 'age', 'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes',
                'tiprel_1mes', 'indresi', 'indext', 'canal_entrada', 'indfall', 'nomprov', 'ind_actividad_cliente',
                'renta', 'segmento', 'month']

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
               'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

ESTIMATORS = {
    "LR": LogisticRegression(),
    "SVM": LinearSVC(),
    "RF": ensemble.RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=10, random_state=2017),
    "GBDT": ensemble.GradientBoostingClassifier(),
    "XGB": xgb()
}


if __name__ == '__main__':
    st = time()
    file_prefix = 'complicateFeatureLabelSubmission'
    # train_X, train_y, test_X, ncodarr = read_data()

    # 计算单个模型，单个模型调参的时候适合。
    # pred_prob_y = single_model_try(train_X, train_y, test_X)
    # result_calc(ncodarr, pred_prob_y, file_prefix)

    # 计算所有模型，调好参数的时候
    # pred_prob_dict = run_model(train_X, train_y, test_X)
    # all_result_calc(ncodarr, pred_prob_dict, file_prefix)
    et = time()
    logger.info('the program cost %f s' % (et - st))

# 清除linux中缓存
# echo 1|sudo tee /proc/sys/vm/drop_caches
