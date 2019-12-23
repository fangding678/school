# -*- coding: utf-8 -*-
# https://www.kaggle.com/apryor6/santander-product-recommendation/detailed-cleaning-visualization-python/notebook
import numpy as np
import pandas as pd

input_path = '/home/ubuntu/Data/'
output_path = '/home/ubuntu/Data/Result/'
#unknown = 'UNKNOWN'


def readAndFillData(all_data=False, filename='train_ver2.csv', writefile='train_fill.csv'):
    unknown = 'UNKNOWN'
    if all_data:
        df = pd.read_csv(input_path + filename, dtype={'sexo': str, 'ind_nuevo': str, 'ult_fec_cli_1t': str,
                                                       'indext': str})
    else:
        df = pd.read_csv(input_path + filename, dtype={"sexo": str, "ind_nuevo": str, "ult_fec_cli_1t": str,
                                                       "indext": str}, nrows=2e6, low_memory=False)
        unique_ids = pd.Series(df.ncodpers.unique())
        limit_people = 10000
        unique_id = unique_ids.sample(n=limit_people)  # 抽样
        df = df[df.ncodpers.isin(unique_id)]
    # tipodom, cod_prov与nomprov的含义一样，删除之
    print('tipodom, cod_prov与nomprov的含义一样，删除之')
    df.drop(['tipodom', 'cod_prov'], axis=1, inplace=True)
    # ult_fec_cli_1t, conyuemp缺失值太多，删除之
    print('ult_fec_cli_1t, conyuemp缺失值太多，删除之')
    df.drop(['ult_fec_cli_1t', 'conyuemp'], axis=1, inplace=True)
    # 转化fecha_dato和fecha_alta格式为日期格式
    print('转化fecha_dato和fecha_alta格式为日期格式')
    df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], format='%Y-%m-%d')
    df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], format='%Y-%m-%d')
    df.age = pd.to_numeric(df.age, errors='coerce')  # age转化为numeric后会有缺失值
    df['month'] = pd.DatetimeIndex(df.fecha_dato).month  # 添加新特征月份
    # 填充age特征缺失值
    print('填充age特征缺失值')
    df.loc[df.age < 18, 'age'] = df.loc[(df.age <= 30) & (df.age >= 18), 'age'].mean(skipna=True)
    df.loc[df.age > 100, 'age'] = df.loc[(df.age > 30) & (df.age <= 100), 'age'].mean(skipna=True)
    df.age.fillna(df.age.mean(), inplace=True)
    df.age = df.age.astype(int)
    # antiguedad是用户等级，填充之后会有缺失值
    print('antiguedad是用户等级，填充之后会有缺失值')
    df.antiguedad = pd.to_numeric(df.antiguedad, errors='coerce')
    df.loc[df.antiguedad.isnull(), 'antiguedad'] = df.antiguedad.min()  # 大部分antiguedad为空用户ind_nuevo都是1，新用户等级低
    df.loc[df.antiguedad < 0, 'antiguedad'] = 0  # 有部分antiguedad值小于0
    # ind_nuevo是否是近期用户，大部分用户都是近6个月用户
    # ind_nuevo缺省值占总量百分之二，填充为2
    print('ind_nuevo是否是近期用户，大部分用户都是近6个月用户')
    df.loc[df.ind_nuevo.isnull(), 'ind_nuevo'] = 2
    # fecha_alta的缺失值，填充fecha_alta的中位数
    print('fecha_alta的缺失值，填充fecha_alta的中位数')
    dates = df.fecha_alta.sort_values().reset_index()
    df.loc[df.fecha_alta.isnull(), 'fecha_alta'] = dates.loc[int(np.median(dates.index.values)), 'fecha_alta']
    # indrel百分之99以上值都是1
    print('indrel绝大多数值都是1')
    df.loc[df.indrel.isnull(), 'indrel'] = 1
    # ind_actividad_cliente有少数缺失值，0、1各占一半，中位数填充
    print('ind_actividad_cliente有少数缺失值，0、1各占一半，中位数填充')
    df.loc[df.ind_actividad_cliente.isnull(), 'ind_actividad_cliente'] = df.ind_actividad_cliente.median()
    # nomprov是指省份,填充缺失值，然后按照省份划分分类
    print('nomprov是指省份,填充缺失值，然后按照省份划分分类')
    df.loc[df.nomprov == 'CORUÑA, A', 'nomprov'] = 'CORUNA, A'
    df.loc[df.nomprov.isnull(), 'nomprov'] = unknown
    # renta是家庭收入，一般而言与省份关系十分密切，按照省份中位数填充，有可能某个省份工资栏都是缺省值，就按照所有工资中位数填充
    # incomes = df.loc[df.renta.notnull(), ['nomprov', 'renta']].groupby('nomprov').agg({'renta': median})
    # incomes.sort_values(by=('renta'), inplace=True)
    # incomes.reset_index(inplace=True)
    # incomes.nomprov = incomes.nomprov.astype("category", categories=[i for i in df.nomprov.unique()], ordered=False)
    print('renta是家庭收入，一般而言与省份关系十分密切，按照省份中位数填充，有可能某个省份工资栏都是缺省值，就按照所有工资中位数填充')
    grouped = df.groupby('nomprov').agg({'renta': lambda x: x.median(skipna=True)}).reset_index()
    income = pd.merge(df, grouped, how='inner', on='nomprov').loc[:, ['nomprov', 'renta_y']]
    income = income.rename(columns={'renta_y': 'renta'}).sort_values('nomprov').reset_index()
    df.sort_values('nomprov', inplace=True)
    df = df.reset_index()
    # grouped按照省中位数填充工资，但是工资都是缺失值的省还不能解决
    print('grouped按照省中位数填充工资，但是工资都是缺失值的省还不能解决')
    df.loc[df.renta.isnull(), 'renta'] = income.loc[df.renta.isnull(), 'renta']
    # 这一步解决工资栏全部都是空的某些省，这种情况在抽样的时候存在，但是所有数据就不会有这样的情况了
    print('这一步解决工资栏全部都是空的某些省')
    df.loc[df.renta.isnull(), 'renta'] = df.loc[df.renta.notnull(), 'renta'].median()
    # 重新调整df的顺序
    print('重新调整df的顺序')
    df.sort_values('fecha_dato', inplace=True)
    # ind_nomina_ult1和ind_nom_pens_ult1两个业务会有千分之一左右的缺省值，两个业务大多数值都是0，填充
    print('ind_nomina_ult1和ind_nom_pens_ult1两个业务会有缺省值，两个业务大多数值都是0，填充')
    df.loc[df.ind_nomina_ult1.isnull(), 'ind_nomina_ult1'] = 0
    df.loc[df.ind_nom_pens_ult1.isnull(), 'ind_nom_pens_ult1'] = 0
    # indrel_1mes是用户类别，缺失值填充为0，百分之99的用户都是1
    print('indrel_1mes是用户类别，缺失值填充为潜在用户')
    map_dict = {1.0: 1, "1.0": 1, "1": 1, 2.0: 2, "2.0": 2, "2": 2, "3.0": 3, 3.0: 3, "3": 3, "4.0": 4, "4": 4, 4.0: 4,
                "P": 5}
    df.indrel_1mes.fillna(0, inplace=True)
    df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x, x))
    df.indrel_1mes = df.indrel_1mes.astype('category')
    # tiprel_1mes填充为unknown
    print("tiprel_1mes填充为unknown")
    df.loc[df.tiprel_1mes.isnull(), 'tiprel_1mes'] = unknown
    # indfall填充为unknown
    print("indfall基本上都是‘N’，填充为unknown")
    df.loc[df.indfall.isnull(), 'indfall'] = unknown
    # ['ind_empleado', 'pais_residencia', 'sexo', 'indresi', 'indext', 'canal_entrada', 'segmento']全部填充
    print("['ind_empleado', 'pais_residencia', 'sexo', 'indresi', 'indext', 'canal_entrada', 'segmento']全部填充为unknown")
    col_fill = ['ind_empleado', 'pais_residencia', 'sexo', 'indresi', 'indext', 'canal_entrada', 'segmento']
    for col in col_fill:
        df.loc[df[col].isnull(), col] = unknown
    print(df.isnull().any())
    # 改变结果属性的类型为int
    print("改变结果属性的类型为int")
    feature_cols = df.iloc[:1, ].filter(regex="ind_+.*ult.*").columns.values
    for col in feature_cols:
        df[col] = df[col].astype(int)
    df.drop(['index'], axis=1, inplace=True)
    # 改变特征值为整型
    df.renta = df.renta.astype(int)
    df.ind_actividad_cliente = df.ind_actividad_cliente.astype(int)
    df.antiguedad = df.antiguedad.astype(int)
    df.indrel = df.indrel.astype(int)
    # 清洗后的数据写入文件中
    print('清洗后的数据写入文件中')
    df.to_csv(input_path + writefile, index=False)


def readAndFillTest(filename='test_ver2.csv', writefile='test_fill.csv'):
    unknown = 'UNKNOWN'
    df = pd.read_csv(input_path + filename, dtype={'sexo': str, 'ind_nuevo': str, 'ult_fec_cli_1t': str, 'indext': str})
    # tipodom, cod_prov与nomprov的含义一样，删除之
    print('tipodom, cod_prov与nomprov的含义一样，删除之')
    df.drop(['tipodom', 'cod_prov'], axis=1, inplace=True)
    # ult_fec_cli_1t, conyuemp缺失值太多，删除之
    print('ult_fec_cli_1t, conyuemp缺失值太多，删除之')
    df.drop(['ult_fec_cli_1t', 'conyuemp'], axis=1, inplace=True)
    # 转化fecha_dato和fecha_alta格式为日期格式
    print('转化fecha_dato和fecha_alta格式为日期格式')
    df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], format='%Y-%m-%d')
    df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], format='%Y-%m-%d')
    df.age = pd.to_numeric(df.age, errors='coerce')  # age转化为numeric后会有缺失值
    df['month'] = pd.DatetimeIndex(df.fecha_dato).month  # 添加新特征月份
    # 填充age特征缺失值,测试数据中age没有缺失值
    print('填充age特征缺失值')
    df.loc[df.age < 18, 'age'] = df.loc[(df.age <= 30) & (df.age >= 18), 'age'].mean(skipna=True)
    df.loc[df.age > 100, 'age'] = df.loc[(df.age > 30) & (df.age <= 100), 'age'].mean(skipna=True)
    df.age.fillna(df.age.mean(), inplace=True)
    df.age = df.age.astype(int)
    # antiguedad是用户等级，填充之后会有缺失值,测试集中不会有缺失值
    print('antiguedad是用户等级，填充之后会有缺失值')
    df.antiguedad = pd.to_numeric(df.antiguedad, errors='coerce')
    df.loc[df.antiguedad.isnull(), 'antiguedad'] = df.antiguedad.min()  # 大部分antiguedad为空用户ind_nuevo都是1，新用户等级低
    df.loc[df.antiguedad < 0, 'antiguedad'] = 0  # 有部分antiguedad值小于0
    # ind_nuevo是否是近期用户，大部分用户都是近6个月用户，测试集中没有缺失值
    print('ind_nuevo是否是近期用户，大部分用户都是近6个月用户')
    df.ind_nuevo = pd.to_numeric(df.ind_nuevo, errors='coerce')
    df.ind_nuevo = df.loc[df.ind_nuevo.isnull(), 'ind_nuevo'] = 1
    # fecha_alta的缺失值，填充fecha_alta的中位数，测试集中没有缺失值
    print('fecha_alta的缺失值，填充fecha_alta的中位数')
    dates = df.fecha_alta.sort_values().reset_index()
    df.loc[df.fecha_alta.isnull(), 'fecha_alta'] = dates.loc[int(np.median(dates.index.values)), 'fecha_alta']
    # indrel绝大多数值都是1，测试集中没有缺失值
    print('indrel绝大多数值都是1')
    df.loc[df.indrel.isnull(), 'indrel'] = 1
    # ind_actividad_cliente有少数缺失值，0、1各占一半，中位数填充，测试集中也没有缺失值
    print('ind_actividad_cliente有少数缺失值，0、1各占一半，中位数填充')
    df.loc[df.ind_actividad_cliente.isnull(), 'ind_actividad_cliente'] = df.ind_actividad_cliente.median()
    # nomprov是指省份,填充缺失值，然后按照省份划分分类
    print('nomprov是指省份,填充缺失值，然后按照省份划分分类')
    df.loc[df.nomprov == 'CORUÑA, A', 'nomprov'] = 'CORUNA, A'
    df.loc[df.nomprov.isnull(), 'nomprov'] = unknown
    # renta是家庭收入，一般而言与省份关系十分密切，按照省份中位数填充，有可能某个省份工资栏都是缺省值，就按照所有工资中位数填充
    # incomes = df.loc[df.renta.notnull(), ['nomprov', 'renta']].groupby('nomprov').agg({'renta': median})
    # incomes.sort_values(by=('renta'), inplace=True)
    # incomes.reset_index(inplace=True)
    # incomes.nomprov = incomes.nomprov.astype("category", categories=[i for i in df.nomprov.unique()], ordered=False)
    df.renta = pd.to_numeric(df.renta, errors="coerce")
    # 上一步之后renta会有很多缺失值，实验中通过训练集的中位数填充
    # grouped = dftrain.groupby('nomprov').agg({'renta': lambda x: x.median(skipna=True)}).reset_index()
    print('renta是家庭收入，一般而言与省份关系十分密切，按照省份中位数填充，有可能某个省份工资栏都是缺省值，就按照所有工资中位数填充')
    grouped = df.groupby('nomprov').agg({'renta': lambda x: x.median(skipna=True)}).reset_index()
    income = pd.merge(df, grouped, how='inner', on='nomprov').loc[:, ['nomprov', 'renta_y']]
    income = income.rename(columns={'renta_y': 'renta'}).sort_values('nomprov').reset_index()
    df.sort_values('nomprov', inplace=True)
    df = df.reset_index()
    # grouped按照省中位数填充工资，但是工资都是缺失值的省还不能解决
    print('grouped按照省中位数填充工资，但是工资都是缺失值的省还不能解决')
    df.loc[df.renta.isnull(), 'renta'] = income.loc[df.renta.isnull(), 'renta']
    # 这一步解决工资栏全部都是空的某些省，填充训练集的中位数
    # df.loc[df.renta.isnull(), 'renta'] = dftrain.loc[dftrain.renta.notnull(), 'renta'].median()
    print('这一步解决工资栏全部都是空的某些省')
    df.loc[df.renta.isnull(), 'renta'] = df.loc[df.renta.notnull(), 'renta'].median()
    # 重新调整df的顺序
    print('重新调整df的顺序')
    df.sort_values('fecha_dato', inplace=True)
    # indrel_1mes是用户类别，缺失值填充为潜在用户
    print('indrel_1mes是用户类别，缺失值填充为潜在用户')
    map_dict = {1.0: "1", "1.0": "1", "1": "1", "3.0": "3", "P": "P", 3.0: "3", 2.0: "2", "3": "3",
                "2.0": "2", "4.0": "4", "4": "4", "2": "2"}
    df.indrel_1mes.fillna(0, inplace=True)
    df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x, x))
    df.indrel_1mes = df.indrel_1mes.astype('category')
    # tiprel_1mes填充为unknown
    print('tiprel_1mes填充为unknown')
    df.loc[df.tiprel_1mes.isnull(), 'tiprel_1mes'] = unknown
    # indfall填充为unknown
    print("indfall填充为unknown")
    df.loc[df.indfall.isnull(), 'indfall'] = unknown
    # ['ind_empleado', 'pais_residencia', 'sexo', 'indresi', 'indext', 'canal_entrada', 'segmento']全部填充
    col_fill = ['ind_empleado', 'pais_residencia', 'sexo', 'indresi', 'indext', 'canal_entrada', 'segmento']
    print("['ind_empleado', 'pais_residencia', 'sexo', 'indresi', 'indext', 'canal_entrada', 'segmento']全部填充")
    for col in col_fill:
        df.loc[df[col].isnull(), col] = unknown
    print(df.isnull().any())
    df.drop(['index'], axis=1, inplace=True)
    # 改变特征值为整型
    df.renta = df.renta.astype(int)
    df.ind_actividad_cliente = df.ind_actividad_cliente.astype(int)
    df.antiguedad = df.antiguedad.astype(int)
    df.indrel = df.indrel.astype(int)
    df.indrel_1mes = df.indrel_1mes.astype(int)
    # 清洗后的数据写入文件中
    print('清洗后的数据写入文件中')
    df.to_csv(input_path + writefile, index=False)


if __name__ == '__main__':
    # filename = 'train_ver2.csv'
    # readAndFillData(True)
    readAndFillTest()
    pass
