# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
from time import time

input_path = '/home/ubuntu/Data/'
output_path = '/home/ubuntu/Data/Result/'


def continuousEncoding1(readfile='train_fill.csv', writefile='feature.csv'):
    df = pd.read_csv(input_path + readfile)
    feeture_cols = df.columns.tolist()
    with open(input_path + writefile, 'w') as f:
        for col in feeture_cols:
            feat = df.loc[:, col].value_counts().sort_values()
            f.write(str(col) + '\n' + str(feat) + '\n\n')


def continuousEncoding2(readfile='train_fill.csv', writefile1='dict_train1.csv', writefile2='dict_train2.csv'):
    df = pd.read_csv(input_path + readfile)
    feature_col = ['fecha_dato', 'ind_empleado', 'pais_residencia', 'sexo', 'indrel', 'tiprel_1mes',
                   'indresi', 'indext', 'canal_entrada', 'indfall', 'nomprov', 'segmento']
    d1 = {}
    d2 = {}
    for col in feature_col:
        feat = df.loc[:, col].value_counts().sort_values()
        d1[col] = dict(zip(list(feat.index), list(feat.values)))
        d2[col] = dict(zip(list(feat.index), range(1, len(feat) + 1)))
    with open(input_path + writefile1, 'w') as f1:
        f1.write(str(d1))
    with open(input_path + writefile2, 'w') as f2:
        f2.write(str(d2))
    '''
    with open(input_path + writefile1, 'w') as j1:
        json.dump(d1, j1)
    with open(input_path + writefile2, 'w') as j2:
        json.dump(d2, j2)
    '''


def featurepro1():
    pass


if __name__ == '__main__':
    st = time()
    readfile = 'test_fill.csv'
    writefile1 = 'dict_test.csv'
    writefile2 = 'dict_test.csv'
    continuousEncoding1()
    continuousEncoding2()
    # continuousEncoding2(readfile, writefile1, writefile2)
    et = time()
    print(str(et - st), 's')
