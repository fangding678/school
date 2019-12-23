# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
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


def calc_last_product():
    df = pd.read_csv(input_path + 'trainLabelEncode.csv')
    df = df[df.fecha_dato == '2016-05-28']
    df = df[['ncodpers'] + target_cols]
    dft = pd.read_csv(input_path + 'testLabelEncode.csv')
    dft = np.array(dft.ncodpers)
    df.set_index(['ncodpers'], inplace=True)
    df = df.loc[dft, :]
    np.save(input_path + 'last_product.npy', df.values)


def simple_feature():
    logger.info('read the data ...')
    df = pd.read_csv(input_path + 'trainLabelEncode.csv')
    df1 = df[df.fecha_dato == '2015-05-28']
    df2 = df[df.fecha_dato == '2015-06-28']
    dft1 = df[df.fecha_dato == '2016-05-28']
    del df
    logger.info('process the training data')
    df1 = df1.set_index('ncodpers')
    df2 = df2.set_index('ncodpers')
    df1 = df1[feature_cols + target_cols]
    df2 = df2[feature_cols + target_cols]
    trainX = []
    trainy = []
    cust_dict = {}
    for index, row in df1.iterrows():
        cust_dict[index] = row.tolist()[18:]
    for index, row in df2.iterrows():
        prev_list = cust_dict.get(index, [0] * 24)
        target_list = row.tolist()[18:]
        new_list = [max(x1 - x2, 0) for (x1, x2) in zip(target_list, prev_list)]
        if sum(new_list) <= 0:
            continue
        for ind, prod in enumerate(new_list):
            if prod > 0:
                trainX.append(row.tolist()[:18] + prev_list)
                trainy.append(ind)
    logger.info('process the testing data')
    dft1 = dft1.set_index('ncodpers')
    dft1 = dft1[feature_cols + target_cols]
    dft2 = pd.read_csv(input_path + 'testLabelEncode.csv')
    dft2 = dft2.set_index('ncodpers')
    dft2 = dft2[feature_cols]
    cust_dict = {}
    testX = []
    for index, row in dft1.iterrows():
        cust_dict[index] = row.tolist()[18:]
    for index, row in dft2.iterrows():
        prev_list = cust_dict.get(index, [0] * 24)
        testX.append(row.tolist() + prev_list)
    usercode = np.array(dft2.index.tolist())
    trainX = np.array(trainX)
    trainy = np.array(trainy)
    testX = np.array(testX)
    logger.info(trainX.shape)   # (45679, 42)
    logger.info(trainy.shape)   # (45679,)
    logger.info(testX.shape)    # (929615, 42)
    logger.info(usercode)
    logger.info("简单特征处理后的数据写入文件中")
    np.save(input_path + "simpleFeatureTrainX.npy", trainX)
    np.save(input_path + "simpleFeatureTrainy.npy", trainy)
    np.save(input_path + "simpleFeatureTestX.npy", testX)
    np.save(input_path + "simpleFeatureUsercode.npy", usercode)


def complicate_feature():
    logger.info('read the data ...')
    df = pd.read_csv(input_path + 'trainLabelEncode.csv')


if __name__ == '__main__':
    st = time()
    logger.info("extract the simple feature...")
    # calc_last_product()
    # simple_feature()
    # complicate_feature()
    et = time()
    logger.info("the program cost %d s" % (et - st))
