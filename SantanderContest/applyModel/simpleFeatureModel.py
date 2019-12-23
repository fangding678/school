# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
    # "SVM": libsvm(),
    "RF": ensemble.RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=10, random_state=2017),
    "GBDT": ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=6, min_samples_leaf=10, random_state=2017),
    #"XGB": xgb()
}


# 筛选出来的训练数据集中只有22个类别，头两个没有出现。
def read_data(trainfileX='simpleFeatureTrainX.npy', trainfiley='simpleFeatureTrainy.npy',
              testfileX='simpleFeatureTestX.npy', usercode='simpleFeatureUsercode.npy'):
    logger.info("load the numpy data...")
    train_X = np.load(input_path + trainfileX)
    test_X = np.load(input_path + testfileX)
    train_y = np.load(input_path + trainfiley)
    ncodarr = np.load(input_path + usercode)
    logger.info("对简要处理后的数据进行归一化操作...")
    lenTX = len(train_X)
    X = np.vstack((train_X, test_X))
    mmScaler = preprocessing.MinMaxScaler()
    X = mmScaler.fit_transform(X)
    train_X = X[:lenTX]
    test_X = X[lenTX:]
    logger.info("返回归一化后的数据...")
    return train_X, train_y, test_X, ncodarr


# 对于多分类问题，模型源码中对所有输出会进行unique操作，以便获取类别总数。unique有排序功能
def run_model(train_X, train_y, test_X):
    pred_prob_dict = dict()
    for name, estimator in ESTIMATORS.items():
        logger.info('Running the Model of... ' + name)
        estimator.fit(train_X, train_y)
        logger.info(name + ' model predict_proba the result...')
        pred_prob_dict[name] = estimator.predict_proba(test_X)
        pred_prob_dict[name] = np.array(pred_prob_dict[name])
        logger.info('the length of predict result: %d' % len(pred_prob_dict[name]))
    return pred_prob_dict


def all_result_calc(ncodarr, pred_prob_dict, file_prefix):
    logger.info('读取最后一个月买产品的字典文件')
    last_arr = np.load(input_path + "last_product.npy")
    last_arr = last_arr[:, 2:]      # 头两种类别在训练数据集中没有
    target = np.array(target_cols[2:])
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


# 尤其要注意，xgb包不是sklearn派系。对于多分类(n)问题，分类标签必须是[0, n)
def run_xgb(train_X, train_y, test_X):
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
    num_rounds = 100
    plst = list(param.items())
    logger.info('training the xgboost model...')
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)
    logger.info('predict the xgboost model...')
    pred_prob_y = model.predict(xgb.DMatrix(test_X))
    pred_prob_y = np.array(pred_prob_y)
    return pred_prob_y


def result_calc_xgb(ncodarr, pred_prob_y, resultfile='simpleFeatureLabelSubmissionXGB1.csv'):
    logger.debug('读取最后一个月买产品的字典文件')
    last_file = "last_product.npy"
    last_arr = np.load(input_path + last_file)
    target = np.array(target_cols)
    logger.info("calc the result of trial model!")
    assert pred_prob_y.shape == last_arr.shape
    pred_prob_y = pred_prob_y - last_arr
    predict_y = pred_prob_y.argsort(axis=1)
    predict_y = np.fliplr(predict_y)[:, :7]
    final_pred = [' '.join(target[pr]) for pr in predict_y]
    dfres = pd.DataFrame({'ncodpers': ncodarr, 'added_products': final_pred})
    dfres = dfres[['ncodpers', 'added_products']]
    logger.debug('结果数据写入文件:' + output_path + resultfile)
    dfres.to_csv(output_path + resultfile, index=False)


if __name__ == '__main__':
    st = time()
    file_prefix = 'simpleFeatureLabelSubmission'
    train_X, train_y, test_X, ncodarr = read_data()

    # 计算所有模型，调好参数的时候
    # pred_prob_dict = run_model(train_X, train_y, test_X)
    # all_result_calc(ncodarr, pred_prob_dict, file_prefix)

    # 计算XGBoost模型
    pred_prob_y = run_xgb(train_X, train_y, test_X)
    result_calc_xgb(ncodarr, pred_prob_y)
    et = time()
    logger.info('the program cost %f s' % (et - st))

# 清除linux中缓存
# echo 1|sudo tee /proc/sys/vm/drop_caches
