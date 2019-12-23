# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

input_path = '/home/ubuntu/Data/'
output_path = '/home/ubuntu/Data/Result/'
local_output_path = 'D:/Codes/Data/SantanderProductRecommendation/Try/'
local_input_path = 'D:/Codes/Data/SantanderProductRecommendation/'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def readjson(dictfile='dict_train2.csv', jsonfile='dict_train2.json'):
    s = ''
    with open(dictfile, 'r') as f:
        s = f.read()
    ds = eval(s)  # 读取出来的s是一个字符串，eval转化为字典
    with open(jsonfile, 'w') as f:
        json.dump(ds, f)  # 写入json文件


def readwritenumpy(infile=local_input_path + 'testLabelEncode.csv', outfile=local_output_path + 'aa.npy'):
    df = pd.read_csv(infile, nrows=100)
    da = df.values
    logging.info(da)
    logging.info(type(da))
    # np.save(outfile, da)
    c = np.load(outfile)
    logger.info(c)


def just_try():
    x0 = np.array([0]*200)
    y0 = np.arange(0, 1, 0.005)
    y00 = np.arange(-1, 1, 0.01)
    x1 = np.arange(-8, 8, 0.01)
    y1 = 1 / (1 + np.exp(-x1))
    y11 = (np.exp(x1) - np.exp(-x1)) / (np.exp(x1) + np.exp(-x1))
    logger.info(x1)
    logger.info(y1)
    plt.figure(figsize=(6, 3))
    plt.subplot(121)
    plt.plot(x0, y0, x1, y1)
    plt.title("sigmoid function")
    plt.subplot(122)
    plt.plot(x0, y00, x1, y11)
    plt.title("tanh function")
    plt.show()


if __name__ == '__main__':
    # readjson()
    # readjson(dictfile='dict_train1.csv', jsonfile='dict_train1.json')
    # readwritenumpy()
    # text = "sdtg"
    # logger.info("{}...%d".format(text) % 12)
    # logger.info("gsdfghds%d" % 13)
    just_try()
    pass
