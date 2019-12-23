from modelMain.engine import *
from modelMain.feature_engineer import *


def make_submission(filename, Y_test, C):
    res_df = pd.DataFrame(C, columns=["ncodpers"])
    with Timer("make submission"):
        res_np = np.argsort(Y_test, axis=1)
        res_np = np.fliplr(res_np)
        Y_ret = res_np.tolist()
        target_cols = np.array(products)
        res_np = res_np[:, :7]
        res_prod = [" ".join(list(target_cols[res])) for res in res_np]
        res_df["added_products"] = res_prod
        res_df.to_csv(path_result + filename, index=False)
    return Y_ret


def train_base_model(str_date, is_file_exist, feature_num=None):
    with Timer("load the training and testing data"):
        train_df = pd.read_pickle(path_input + "data_train.pickle")
        test_df = pd.read_pickle(path_input + "data_test.pickle")
        with open(path_input + "importance_features.pickle", "rb") as fr:
            (features, prod_features) = pickle.load(fr)

    if feature_num is not None:
        logger.info("select the feature")
        features = features[:feature_num]

    C = np.array(test_df.ncodpers)
    Y_prev = test_df.as_matrix(columns=prod_features)

    with Timer("train the LR-RF-GBDT"):
        Y_test_dict = base_model(train_df, test_df, features, str_date, is_file_exist)
    logger.info("make submission of LR-RF-GBDT")
    for name, Y_test in Y_test_dict.items():
        test_add_list = make_submission("features%d_base-%s.%s.csv" % (len(features), str_date, name), Y_test - Y_prev,
                                        C)


def train_model(str_date, is_file_exist=False, feature_num=None):
    with Timer("load the training and testing data"):
        if str_date == "2016-05-28":
            train_df = pd.read_pickle(path_input + "data_train_cv.pickle")
            test_df = pd.read_pickle(path_input + "data_test_cv.pickle")
            with open(path_input + "result_add_list.pickle", "rb") as fr:
                test_add_list = pickle.load(fr)
        elif str_date == "2016-06-28":
            train_df = pd.read_pickle(path_input + "data_train.pickle")
            test_df = pd.read_pickle(path_input + "data_test.pickle")
        with open(path_input + "importance_features.pickle", "rb") as fr:
            (features, prod_features) = pickle.load(fr)

    if feature_num is not None:
        logger.info("select the feature")
        features = features[:feature_num]

    logger.info("select the feature:")
    features = features[:feature_num]

    if str_date == "2016-05-28":
        logger.info("calcuate the max probability of map@7")
        max_map7 = mapk(test_add_list, test_add_list)
        map7coef = float(len(test_add_list)) / float(sum([int(bool(a)) for a in test_add_list]))
        logger.info("max map@7 %s---%s---%s" % (str_date, max_map7, max_map7 * map7coef))

    logger.info("random split training data into train and validate")
    mask = np.random.rand(len(train_df)) < 0.8
    XY_train = train_df[mask]
    XY_validate = train_df[~mask]
    C = np.array(test_df.ncodpers)
    Y_prev = test_df.as_matrix(columns=prod_features)

    with Timer("XGBoost"):
        logger.info("running the XGBoost model")
        Y_test_xgb = xgboost(XY_train, XY_validate, test_df, features, train_df, str_date, is_file_exist)
        logger.info("make submission of XGBoost")
        test_add_list_xgboost = make_submission("features%d_%s.xgboost.csv" % (len(features), str_date),
                                                Y_test_xgb - Y_prev, C)
        if str_date == "2016-05-28":
            logger.info("XGBoost calcuate the cross validate map@7")
            max7xgboost = mapk(test_add_list, test_add_list_xgboost)
            logger.info("XGBoost map@7 %s---%s---%s" % (str_date, max7xgboost, max7xgboost * map7coef))

    with Timer("LightGBM"):
        logger.info("running the LightGBM model")
        Y_test_lgbm = lightgbm(XY_train, XY_validate, test_df, features, train_df, str_date, is_file_exist)
        logger.info("make submission of LightGBM")
        test_add_list_lightgbm = make_submission("features%d_%s.lightgbm.csv" % (len(features), str_date),
                                                 Y_test_lgbm - Y_prev, C)
        if str_date == "2016-05-28":
            logger.info("LightGBM calcuate the cross validate map@7")
            map7lightgbm = mapk(test_add_list, test_add_list_lightgbm)
            logger.info("LightGBM map@7 %s---%s---%s" % (str_date, map7lightgbm, map7lightgbm * map7coef))

    with Timer("XGBoost+LightGBM"):
        logger.info("mix the XGBoost and LightGBM")
        Y_test = np.sqrt(np.multiply(Y_test_xgb, Y_test_lgbm))
        test_add_list_xgb_gbm = make_submission("features%d_%s.xgboost-lightgbm.csv" % (len(features), str_date),
                                                Y_test - Y_prev, C)
        if str_date == "2016-05-28":
            map7xgb_gbm = mapk(test_add_list, test_add_list_xgb_gbm)
            logger.info("XGBoost+LightGBM map@7 %s---%s---%s" % (str_date, map7xgb_gbm, map7xgb_gbm * map7coef))


if __name__ == '__main__':
    st = time.time()
    # all_df, features, prod_features = complicate_feature_engineer(True)
    # train_prepare(str_date="2016-06-28", is_data_exist=True)

    # train_base_model(str_date="2016-06-28", is_file_exist=False, feature_num=32)
    # train_model(str_date="2016-05-28", is_file_exist=False, feature_num=32)
    # train_model(str_date="2016-06-28", is_file_exist=False, feature_num=32)

    # train_base_model(str_date="2016-06-28", is_file_exist=False, feature_num=80)
    # train_model(str_date="2016-05-28", is_file_exist=False, feature_num=80)
    # train_model(str_date="2016-06-28", is_file_exist=False, feature_num=80)

    train_base_model(str_date="2016-06-28", is_file_exist=False)
    train_model(str_date="2016-05-28", is_file_exist=False)
    train_model(str_date="2016-06-28", is_file_exist=False)
    et = time.time()
    logger.info("The total time of program cost is %f" % (et - st))


# 清除linux中缓存
# echo 1|sudo tee /proc/sys/vm/drop_caches
