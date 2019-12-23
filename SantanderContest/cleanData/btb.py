# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from time import time

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
               'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
input_path = '/home/ubuntu/Data/'
output_path = '/home/ubuntu/Data/Result/'


def sortByProduct():
    df = pd.read_csv(input_path + 'train_fill.csv', usecols=target_cols)
    product_list = list(df.sum(axis=0).sort_values(ascending=False).index)
    return product_list


def simpleBTB(product_list):
    dfs = pd.read_csv(output_path + 'sample_submission.csv')
    s = ' '.join(product_list[:7])
    dfs.added_products = s
    dfs.to_csv(output_path + 'submissionSimpleBTB1.csv', index=False)


def filterProduct(product_list, ll):
    s = []
    c = 0
    for i, v in enumerate(ll):
        if c >= 7:
            break
        if v != 0:
            continue
        s.append(product_list[i])
        c += 1
    return ' '.join(s)


def complicateBTB(product_list):
    df = pd.read_csv(input_path + 'train_fill.csv', usecols=['ncodpers', 'fecha_dato'] + target_cols)  # 读取训练集数据
    df = df[df.fecha_dato == '2016-05-28']  # 萃取最后一个月的数据
    df = df[['ncodpers'] + product_list]  # 对数据列进行重排
    df = df.set_index('ncodpers')  # 指定用户编号为索引
    dfs = pd.read_csv(output_path + 'sample_submission.csv')  # 读取要预测用户的编号
    dfs = dfs.set_index('ncodpers')
    # 分析得知预测929615个用户，在2016年5月中都有预测的用户
    for ind in dfs.index.tolist():
        ll = df.loc[ind].tolist()
        dfs.loc[ind, 'added_products'] = filterProduct(product_list, ll)
    dfs.to_csv(output_path + "submissionComplicateBTB2.csv", index_label='ncodpers')


if __name__ == '__main__':
    st = time()
    product_list = sortByProduct()
    # simpleBTB(product_list)
    complicateBTB(product_list)
    et = time()
    print('the program cost %f s' % (et-st))
