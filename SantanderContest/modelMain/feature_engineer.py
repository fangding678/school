import pandas as pd
from sklearn.preprocessing import LabelEncoder
from modelMain.utils import *
import pickle
import math

transformers = {}


def label_code(df, features, name):
    df[name] = df[name].astype('str')
    if name in transformers:
        df[name] = transformers[name].transform(df[name])
    else:
        transformers[name] = LabelEncoder()
        df[name] = transformers[name].fit_transform(df[name])
    features.append(name)


def one_hot(df, features, name, name_dict):
    for k, v in name_dict.items():
        new_name = "%s_%s" % (name, v)
        df[new_name] = df[name].map(lambda x: 1 if x == k else 0).astype(np.int8)
        features.append(new_name)
    df.drop(name, axis=1, inplace=True)


def load_data(train_file="train_fill.csv", test_file="test_fill.csv"):
    with Timer('read origin train file'):
        train_df = pd.read_csv(path_input + train_file)
    with Timer('read origin test file'):
        test_df = pd.read_csv(path_input + test_file)
    with Timer('concat train data frame and test data frame'):
        all_df = pd.concat([train_df, test_df])
        all_df = all_df[train_df.columns]
    with Timer('fill products with zero'):
        all_df[list(products)] = all_df[list(products)].fillna(0.0).astype(np.int8)
    return all_df


def origin_feature_transforms(all_df):
    features = []
    with Timer('origin features transforms'):
        logger.info("transform the explicit origin features")
        logger.info("label encode")
        label_code(all_df, features, "pais_residencia")
        all_df["pais_residencia"] = all_df["pais_residencia"].astype(np.int16)
        label_code(all_df, features, "canal_entrada")
        all_df["canal_entrada"] = all_df["canal_entrada"].astype(np.int16)
        label_code(all_df, features, "nomprov")
        all_df["nomprov"] = all_df["nomprov"].astype(np.int16)

        all_df["age"] = all_df["age"].astype(np.int16)
        features.append('age')
        all_df["antiguedad"] = all_df["antiguedad"].astype(np.int16)
        features.append('antiguedad')
        all_df["indrel_1mes"] = all_df["indrel_1mes"].astype(np.int8)
        features.append("indrel_1mes")
        all_df["ind_actividad_cliente"] = all_df["ind_actividad_cliente"].astype(np.int8)
        features.append("ind_actividad_cliente")
        all_df['renta'] = all_df['renta'].map(math.log)
        features.append('renta')

        logger.info("onehot encode")
        one_hot(all_df, features, "ind_empleado", {"A": "a", "B": "b", "F": "f", "N": "n"})
        one_hot(all_df, features, "sexo", {"V": "v", "H": "h"})
        one_hot(all_df, features, "ind_nuevo", {1: "new"})
        one_hot(all_df, features, "indrel", {1: "1", 99: "99"})
        one_hot(all_df, features, "tiprel_1mes", {"A": "a", "I": 'i', "P": "p", "R": "r"})
        one_hot(all_df, features, "indresi", {"N": "n"})
        one_hot(all_df, features, "indext", {"S": "s"})
        one_hot(all_df, features, "indfall", {"S": "s"})
        one_hot(all_df, features, "segmento",
                {"01 - TOP": "top", "02 - PARTICULARES": "particulares", "03 - UNIVERSITARIO": "universitario"})

        logger.info("add the implicit date features")
        all_df["fecha_dato_year"] = all_df["fecha_dato"].map(lambda x: float(x.split("-")[0])).astype(np.int16)
        features.append("fecha_dato_year")
        all_df["fecha_dato_month"] = all_df["fecha_dato"].map(lambda x: float(x.split("-")[1])).astype(np.int8)
        features.append("fecha_dato_month")
        all_df["fecha_alta_year"] = all_df["fecha_alta"].map(lambda x: float(x.split("-")[0])).astype(np.int16)
        features.append("fecha_alta_year")
        all_df["fecha_alta_month"] = all_df["fecha_alta"].map(lambda x: float(x.split("-")[1])).astype(np.int8)
        features.append("fecha_alta_month")
        all_df["dato_minus_alta"] = all_df["fecha_dato"].map(date_to_int2) - all_df["fecha_alta"].map(date_to_int2)
        all_df["dato_minus_alta"] = all_df["dato_minus_alta"].astype(np.int16)
        features.append("dato_minus_alta")
        all_df['int_date'] = all_df["fecha_dato"].map(date_to_int1).astype(np.int8)
        # features.append('int_date')

        logger.info("delete the origin date information")
        all_df.drop(["fecha_dato", "fecha_alta"], axis=1, inplace=True)

    return all_df, tuple(features)


def make_prev_df(all_df, step):
    with Timer("make prev_%d DF" % step):
        prev_df = pd.DataFrame()
        prev_df['ncodpers'] = all_df['ncodpers']
        prev_df["int_date"] = all_df["int_date"].map(lambda x: x+step).astype(np.int8)
        prod_features = ["%s_prev_%d" % (product, step) for product in products]
        for prod, prev in zip(products, prod_features):
            prev_df[prev] = all_df[prod]
    return prev_df, tuple(prod_features)


def join_with_df(all_df, prev_df, how):
    with Timer("join %s " % how):
        assert set(all_df.columns.values.tolist()) & set(prev_df.columns.values.tolist()) == set(["ncodpers", "int_date"])
        logger.info("before join the length of all_df: %d" % len(all_df))
        all_df = all_df.merge(prev_df, on=["ncodpers", "int_date"], how=how)
        for f in set(prev_df.columns.values.tolist()) - set(["ncodpers", "int_date"]):
            all_df[f] = all_df[f].astype(np.float16)
        logger.info("after join the length of all_df: %d" % len(all_df))
    return all_df


def construct_feature(all_df, features):
    prev_dfs = []
    prod_features = None
    logger.info("construct the new features")

    logger.info("make the previous dataframe as features and prepare for next step")
    for step in range(1, 6):
        prev_df1, prod_features1 = make_prev_df(all_df, step)
        prev_dfs.append(prev_df1)
        if step == 1 or step == 2:
            features += prod_features1
        if step == 1:
            prod_features = prod_features1

    logger.info("join the dataframe")
    for i, prev_df in enumerate(prev_dfs):
        with Timer("join all dataframe with prev_%d" % (i+1)):
            how = "inner" if i == 0 else "left"
            all_df = join_with_df(all_df, prev_df, how=how)

    # with Timer("write the temp train and test to file"):
    #     all_df.to_pickle(path_input + "all_data_1.pickle")
    #     with open(path_input + "all_feature_1.pickle", "wb") as fw:
    #         pickle.dump((features, prod_features), fw)

    logger.info("add the features of product std")
    for prod in products:
        logger.info(prod + "---std")
        for b, e in [(1, 3), (1, 5), (2, 5)]:
            prods = ["%s_prev_%d" % (prod, i) for i in range(b, e+1)]
            mp_df = all_df.as_matrix(columns=prods)

            stdf = "%s_std_%s_%s" % (prod, b, e)
            all_df[stdf] = np.nanstd(mp_df, axis=1)

            features += (stdf,)

    logger.info("add the features of product maxmin")
    for prod in products:
        logger.info(prod + "---maxmin")
        for b, e in [(2, 3), (2, 5)]:
            prods = ["%s_prev_%d" % (prod, i) for i in range(b, e+1)]
            mp_df = all_df.as_matrix(columns=prods)

            maxf = "%s_max_%s_%s" % (prod, b, e)
            all_df[maxf] = np.nanmax(mp_df, axis=1).astype(np.int8)

            minf = "%s_min_%s_%s" % (prod, b, e)
            all_df[minf] = np.nanmin(mp_df, axis=1).astype(np.int8)

            features += (maxf, minf,)

    with Timer("remove the unused columns and write data to file"):
        columns = ["ncodpers", "int_date"] + list(products) + list(features)
        assert len(columns) == len(set(columns))
        all_df = all_df[columns]

    with Timer("write the processed data to file"):
        all_df.to_pcikle(path_input + "all_data.pickle")
        with open(path_input + "all_feature.pickle", "wb") as fw:
            pickle.dump((features, prod_features), fw)

    return all_df, features, prod_features


def construct_feature1():
    with Timer("load the temp train and test to file"):
        all_df = pd.read_pickle(path_input + "all_data_1.pickle")
        (features, prod_features) = pickle.load(open(path_input + "all_feature_1.pickle", "rb"))

    logger.info("add the features of product std")
    for prod in products:
        logger.info(prod + "---std")
        for b, e in [(1, 3), (1, 5), (2, 5)]:
            prods = ["%s_prev_%d" % (prod, i) for i in range(b, e+1)]
            mp_df = all_df.as_matrix(columns=prods)

            stdf = "%s_std_%s_%s" % (prod, b, e)
            all_df[stdf] = np.nanstd(mp_df, axis=1)

            features += (stdf,)

    logger.info("add the features of product maxmin")
    for prod in products:
        logger.info(prod + "---maxmin")
        for b, e in [(2, 3), (2, 5)]:
            prods = ["%s_prev_%d" % (prod, i) for i in range(b, e+1)]
            mp_df = all_df.as_matrix(columns=prods)

            maxf = "%s_max_%s_%s" % (prod, b, e)
            all_df[maxf] = np.nanmax(mp_df, axis=1).astype(np.int8)

            minf = "%s_min_%s_%s" % (prod, b, e)
            all_df[minf] = np.nanmin(mp_df, axis=1).astype(np.int8)

            features += (maxf, minf,)

    with Timer("remove the unused columns and write data to file"):
        columns = ["ncodpers", "int_date"] + list(products) + list(features)
        assert len(columns) == len(set(columns))
        all_df = all_df[columns]

    logger.info("fill the nan of all data")
    all_df.fillna(0, inplace=True)

    with Timer("write the processed data to file"):
        all_df.to_pickle(path_input + "all_data.pickle")
        # features += ("weights",)
        with open(path_input + "all_features.pickle", "wb") as fw:
            pickle.dump((features, prod_features), fw)

    return all_df, features, prod_features


def complicate_feature_engineer(is_data_exist):
    # return construct_feature1()
    if is_data_exist:
        all_df = pd.read_pickle(path_input + "all_data.pickle")
        with open(path_input + "all_features.pickle", "rb") as fr:
            (features, prod_features) = pickle.load(fr)
        return all_df, features, prod_features
    all_df = load_data()
    all_df, features = origin_feature_transforms(all_df)
    return construct_feature(all_df, features)


def train_prepare(str_date, is_data_exist):
    all_df = pd.read_pickle(path_input + "all_data.pickle")
    if is_data_exist:
        XY = pd.read_pickle(path_input + "train_data.pickle")
        test_df = pd.read_pickle(path_input + "test_data.pickle")
        return XY, test_df
    logger.info("split the train set and test set")
    test_date = date_to_int1(str_date)
    train_df = all_df[all_df.int_date < test_date]
    test_df = all_df[all_df.int_date == test_date]
    test_cv_df = all_df[all_df.int_date == test_date - 1]

    with Timer("select the useful data to training"):
        X = []
        Y = []
        for i, prod in enumerate(products):
            prev = prod + "_prev_1"
            prX = train_df[(train_df[prod] == 1) & (train_df[prev] == 0)]
            prY = np.zeros(prX.shape[0], np.int8) + i
            X.append(prX)
            Y.append(prY)
            logger.info(prod)
            logger.info(prX.shape)

        XY = pd.concat(X)
        Y = np.hstack(Y)
        XY["y"] = Y
    logger.info("the length of train data: %d" % len(XY))

    logger.info("release the memory")
    del all_df
    del train_df

    with Timer("add the weights features of train_set and test_set"):
        XY["ncodepers_int_date"] = XY["ncodpers"].astype(str) + "_" + XY["int_date"].astype(str)
        uniqs, counts = np.unique(XY["ncodepers_int_date"], return_counts=True)
        weights = np.exp(1.0 / counts - 1)
        logger.info(np.unique(counts, return_counts=True))
        logger.info(np.unique(weights, return_counts=True))
        wdf = pd.DataFrame()
        wdf["ncodepers_int_date"] = uniqs
        wdf["weights"] = weights
        XY = XY.merge(wdf, on="ncodepers_int_date")
        XY.drop(["ncodepers_int_date"], axis=1, inplace=True)
        XY_cv = XY[XY.int_date < test_date - 1]
        test_df["weights"] = np.ones(len(test_df), dtype=np.int8)
        test_cv_df["weights"] = np.ones(len(test_cv_df), dtype=np.int8)
    logger.info("the shape of XY is following")
    logger.info(XY.shape)

    with Timer("prepare the cross validate test data"):
        test_add_list = [list() for i in range(len(test_cv_df))]
        for prod in products:
            prev = prod + "_prev_1"
            padd = prod + "_add"
            test_cv_df[padd] = test_cv_df[prod] - test_cv_df[prev]
        test_add_mat = test_cv_df.as_matrix(columns=[prod + "_add" for prod in products])
        assert test_add_mat.shape == (len(test_cv_df), len(products))
        for c in range(len(test_cv_df)):
            for p in range(len(products)):
                if test_add_mat[c, p] > 0:
                    test_add_list[c].append(p)

    with Timer("write the train_data and test_data"):
        XY.to_pickle(path_input + "data_train.pickle")
        test_df.to_pickle(path_input + "data_test.pickle")
        XY_cv.to_pickle(path_input + "data_train_cv.pickle")
        test_cv_df.to_pickle(path_input + "data_test_cv.pickle")
        with open(path_input + "result_add_list.pickle", "wb") as fw:
            pickle.dump(test_add_list, fw)




