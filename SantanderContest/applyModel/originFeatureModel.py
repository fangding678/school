# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import ensemble
import xgboost as xgb
from time import time
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
    # "LR": LogisticRegression(),
    # "SVM": LinearSVC(),
    "RF": ensemble.RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=10, random_state=2017),
    # "GBDT": ensemble.GradientBoostingClassifier(),
    # "XGB": xgb()
}


def read_data(trainfile='trainLabelEncode.csv', testfile='testLabelEncode.csv'):
    logger.info('读取文件中...')
    df = pd.read_csv(input_path + trainfile, index_col='ncodpers')
    dfX = df[feature_cols]
    dfy = df[target_cols].astype('float16')
    dft = pd.read_csv(input_path + testfile, usecols=['ncodpers'] + feature_cols)
    logger.info('文件读取完毕...')
    train_X = np.array(dfX.values)
    train_y = np.array(dfy.values)
    test_X = np.array(dft[feature_cols].values)
    ncodarr = dft['ncodpers'].tolist()
    logger.info("对训练集和测试集特征进行归一化处理")
    lenTX = len(train_X)
    X = np.vstack((train_X, test_X))
    mmScaler = preprocessing.MinMaxScaler()
    X = mmScaler.fit_transform(X)
    train_X = X[:lenTX]
    test_X = X[lenTX:]
    logger.info('logger.info train_X, input any string:')
    logger.info(train_X.shape)
    logger.info('logger.info train_y, input any string:')
    logger.info(train_y.shape)
    logger.info('logger.info test_X, input any string:')
    logger.info(test_X.shape)
    logger.info('logger.info dfncod, input any string:')
    logger.info(len(ncodarr))
    return train_X, train_y, test_X, ncodarr


def run_model(train_X, train_y, test_X):
    logger.info('Running the Model...')
    pred_prob_dict = dict()
    for name, estimator in ESTIMATORS.items():
        logger.info('fit the model of ' + name)
        estimator.fit(train_X, train_y)
        logger.info(name + 'model predict_proba the result...')
        pred_prob_dict[name] = estimator.predict_proba(test_X)
        logger.info(name + 'model print the predict_proba result')
        logger.info(len(pred_prob_dict[name]))
        logger.info(name + 'model print the change of predict_proba result')
        pred_prob_dict[name] = np.array(pred_prob_dict[name])[:, :, 1].T
        logger.info(len(pred_prob_dict[name]))
    return pred_prob_dict


def single_model_try(train_X, train_y, test_X):
    logger.info('Running the Model of RF...')
    model = LogisticRegression()
    model.fit(train_X, train_y)
    logger.info('modelRF predict_proba the result...')
    predict_prob_y = model.predict_proba(test_X)
    logger.info('modelRF print the predict_proba result')
    logger.info(predict_prob_y.shape)
    logger.info('modelRF print the change of predict_proba result')
    pred_prob_y = np.array(predict_prob_y)[:, :, 1].T
    logger.info(pred_prob_y.shape)
    return pred_prob_y


def all_result_calc(ncodarr, pred_prob_dict, file_prefix):
    logger.debug('读取最后一个月买产品的字典文件')
    last_file = "last_product.npy"
    last_arr = np.load(input_path + last_file)
    target = np.array(target_cols)
    for name, pred in pred_prob_dict.items():
        logger.info('calc the result ' + name)
        assert pred.shape == last_arr.shape
        preds = pred - last_arr
        predict_y = preds.argsort(axis=1)
        predict_y = np.fliplr(predict_y)[:, :7]
        final_pred = [' '.join(target[pr]) for pr in predict_y]
        dfres = pd.DataFrame({'ncodpers': ncodarr, 'added_products': final_pred})
        dfres = dfres[['ncodpers', 'added_products']]
        logger.debug('结果数据写入文件:' + output_path + file_prefix + name + '.csv')
        dfres.to_csv(output_path + file_prefix + name + '.csv', index=False)


def result_calc(ncodarr, pred_prob_y, resultfile='submissionLR.csv'):
    logger.debug('读取最后一个月买产品的字典文件')
    last_file = "last_product.npy"
    last_arr = np.load(input_path + last_file)
    logger.info("calc the result of trial model!")
    assert pred_prob_y.shape == last_arr.shape
    pred_prob_y = pred_prob_y - last_arr
    predict_y = pred_prob_y.argsort(axis=1)
    predict_y = np.fliplr(predict_y)[:, :7]
    target = np.array(target_cols)
    final_pred = [' '.join(target[pr]) for pr in predict_y]
    dfres = pd.DataFrame({'ncodpers': ncodarr, 'added_products': final_pred})
    dfres = dfres[['ncodpers', 'added_products']]
    logger.debug('结果数据写入文件:' + output_path + resultfile)
    dfres.to_csv(output_path + resultfile, index=False)


if __name__ == '__main__':
    st = time()
    file_prefix = 'originFeatureLabelSubmission'
    train_X, train_y, test_X, ncodarr = read_data()

    # 计算单个模型，单个模型调参的时候适合。
    # pred_prob_y = single_model_try(train_X, train_y, test_X)
    # result_calc(ncodarr, pred_prob_y, file_prefix)

    # 计算所有模型，调好参数的时候
    pred_prob_dict = run_model(train_X, train_y, test_X)
    all_result_calc(ncodarr, pred_prob_dict, file_prefix)
    et = time()
    logger.info('the program cost %f s' % (et - st))


# 清除linux中缓存
# echo 1|sudo tee /proc/sys/vm/drop_caches


