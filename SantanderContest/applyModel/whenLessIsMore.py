# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from time import time
import logging

import xgboost as xgb

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
    logger.info('read the data ...')
    df = pd.read_csv(input_path + 'trainLabelEncode.csv')
    df1 = df[df.fecha_dato == '2015-05-28']
    df2 = df[df.fecha_dato == '2015-06-28']
    dft1 = df[df.fecha_dato == '2016-05-28']
    del df
    logger.info('process the training data')
    df1 = df1[feature_cols+target_cols]
    df2 = df2[feature_cols+target_cols]
    dft2 = pd.read_csv(input_path + 'testLabelEncode.csv')
    trainX = []
    trainy = []
    cust_dict = {}
    df1 = df1.set_index('ncodpers')
    df2 = df2.set_index('ncodpers')
    for index, row in df1.iterrows():
        cust_dict[index] = row.tolist()[18:]
    for index, row in df2.iterrows():
        prev_list = cust_dict.get(index, [0]*24)
        target_list = row.tolist()[18:]
        new_list = [max(x1-x2, 0) for (x1, x2) in zip(target_list, prev_list)]
        if sum(new_list) <= 0:
            continue
        for ind, prod in enumerate(new_list):
            if prod > 0:
                trainX.append(row.tolist()[:18] + prev_list)
                trainy.append(ind)
    logger.info('process the testing data')
    dft1 = dft1[feature_cols+target_cols]
    dft2 = dft2[feature_cols]
    dft1 = dft1.set_index('ncodpers')
    dft2 = dft2.set_index('ncodpers')
    cust_dict = {}
    testX = []
    for index, row in dft1.iterrows():
        cust_dict[index] = row.tolist()[18:]
    for index, row in dft2.iterrows():
        prev_list = cust_dict.get(index, [0]*24)
        testX.append(row.tolist() + prev_list)
    usercode = dft2.index.tolist()
    trainX = np.array(trainX)
    trainy = np.array(trainy)
    testX = np.array(testX)
    logger.info(trainX.shape)
    logger.info(trainy.shape)
    logger.info(testX.shape)
    logger.info(usercode)
    return trainX, trainy, testX, usercode


def run_xgboost(trainX, trainy, testX):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 8
    param['silent'] = 1
    param['num_class'] = 24
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = 2017
    num_rounds = 50
    plst = list(param.items())
    xgtrain = xgb.DMatrix(trainX, label=trainy)
    model = xgb.train(plst, xgtrain, num_rounds)
    preds = model.predict(xgb.DMatrix(testX))
    return preds


def calc_result(preds):
    ntarget_cols = np.array(target_cols)
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)[:, :7]
    logger.info(preds)
    testid = np.array(pd.read_csv(input_path + 'testLabelEncode.csv', usecols=['ncodpers'])['ncodpers'])
    final_preds = [" ".join(list(ntarget_cols[pred])) for pred in preds]
    result_df = pd.DataFrame({'ncodpers': testid, 'added_products': final_preds})
    logger.info(result_df)
    result_df = result_df[['ncodpers', 'added_products']]
    result_df.to_csv(output_path + "submission_xgb.csv", index=False)


if __name__ == '__main__':
    st = time()
    trainX, trainy, testX, usercode = process_data()
    st2 = time()
    logger.info('the program process data used %d s' % (st2 - st))
    preds = run_xgboost(trainX, trainy, testX)
    logger.info('the program running the model used %d s' % (time() - st2))
    calc_result(preds)
    logger.info('the program is running %d s' % (time() - st))
