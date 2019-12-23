import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

input_path = 'D:/Codes/Data/SantanderProductRecommendation/'
output_path = 'D:/Codes/Data/SantanderProductRecommendation/Picture/'


def preprocess_file():
    dfa = pd.read_csv(input_path + 'age.csv', header=None, names=['age', 'ageCount'])
    dfa = dfa.sort_values(by='age')
    dfa.to_csv(input_path + 'age1.csv', index=False)
    dfr = pd.read_csv(input_path + 'renta.csv', header=None, names=['salary'])
    dfr.sort_values(by='salary', inplace=True)
    dfr.to_csv(input_path + 'renta1.csv', index=False)


def paint_age():
    df = pd.read_csv(input_path + 'age.csv')
    df.plot(kind='bar', x='age', y='ageCount')
    # df.age.hist(bins=10)
    plt.legend(['count'], loc='best')
    plt.title('the count of age')
    plt.savefig(output_path + 'age.jpg')
    plt.show()
    print(df)


def paint_renta():
    df = pd.read_csv(input_path + 'renta.csv')
    # lcut = range(10000, 1000000, 10000)
    df.loc[df.salary < 10000, 'salary'] = 10000
    df.loc[df.salary > 400000, 'salary'] = 400000
    df.hist(bins=20)
    plt.title("the salary distribution")
    plt.savefig(output_path + 'salary10000-400000-20.jpg')
    plt.show()


def readjson(dictfile='dict_train2.csv', jsonfile='dict_train2.json'):
    s = ''
    with open(input_path + dictfile, 'r') as f:
        s = f.read()
    ds = eval(s)  # 读取出来的s是一个字符串，eval转化为字典
    with open(input_path + jsonfile, 'w') as f:
        json.dump(ds, f)  # 写入json文件



if __name__ == '__main__':
    # preprocess_file()
    # paint_age()
    # paint_renta()
    # readjson()
    # readjson(dictfile='dict_train1.csv', jsonfile='dict_train1.json')
    pass
