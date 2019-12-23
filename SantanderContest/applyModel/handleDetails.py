import numpy as np
import pandas as pd
import pickle
import logging

input_path = '/home/ubuntu/Data/'
output_path = '/home/ubuntu/Data/Result/'


def build_last_product_dict(trainfile='trainLabelEncode.csv', dictfile='last_product.pickle'):
    df = pd.read_csv(input_path + trainfile)
    target_cols = np.array(df.columns[1:2].tolist() + df.columns[20:44].tolist())
    df = df.loc[df.fecha_dato == '2016-05-28', target_cols]
    last_dict = {}
    df.set_index('ncodpers', inplace=True)
    target_cols = target_cols[1:]
    # 建立字典，删选出已经购买的产品
    for ind, row in df.iterrows():
        # 注意，字典key值必须是字符串类型才能读入文件。这就尴尬了。读取文件解析的时候也蛋疼了
        last_dict[ind] = set(target_cols[np.array(row == 1)])
    # 字典写入二进制文件
    pickle.dump(last_dict, open(input_path + dictfile, 'wb'))


if __name__ == '__main__':
    build_last_product_dict()


