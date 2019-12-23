import xgboost as xgb
import pickle
import lightgbm as lgbm
from modelMain.utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def base_model(train_df, test_df, features, str_date, is_file_exist):
    classifiers = {"Logistic": LogisticRegression(n_jobs=8),
                   "RandomForest": RandomForestClassifier(max_depth=10, n_jobs=8),
                   "GBDT": GradientBoostingClassifier()}
    model_dict = {}
    Y_test_dict = {}
    if not is_file_exist:
        with Timer("training the base model LR-RF-GBDT"):
            for name, clf in classifiers.items():
                with Timer("training the %s model" % name):
                    clf.fit(train_df[list(features)], train_df["y"], sample_weight=np.array(train_df["weights"]))
                model_dict[name] = clf
        logger.info("write all base model to file")
        with open(path_model + "features%d_base%s.all.model.pickle" % (len(features), str_date), "wb") as fw:
            pickle.dump(model_dict, fw)
    else:
        with Timer("load all base model"):
            model_dict = pickle.load(
                open(path_model + "features%d_base%s.all.model.pickle" % (len(features), str_date), "rb"))

    for name, clf in model_dict.items():
        logger.info("%s model to predict" % name)
        Y_test_dict[name] = clf.predict_proba(test_df[list(features)])

    return Y_test_dict


def xgboost(XY_train, XY_validate, test_df, features, XY_all, str_date, is_file_exist):
    param = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'min_child_weight': 10,
        'max_depth': 8,
        'silent': 1,
        'nthread': 8,
        'eval_metric': 'mlogloss',
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.9,
        'num_class': len(products),
    }

    if not is_file_exist:
        logger.info("prepare train to split validate")
        X_train = XY_train.as_matrix(columns=features)
        Y_train = XY_train.as_matrix(columns=["y"])
        W_train = XY_train.as_matrix(columns=["weights"])
        train = xgb.DMatrix(X_train, label=Y_train, feature_names=features, weight=W_train)

        X_validate = XY_validate.as_matrix(columns=features)
        Y_validate = XY_validate.as_matrix(columns=["y"])
        W_validate = XY_validate.as_matrix(columns=["weights"])
        validate = xgb.DMatrix(X_validate, label=Y_validate, feature_names=features, weight=W_validate)

        logger.info(param)
        evallist = [(train, 'train'), (validate, 'eval')]
        with Timer("XGBoost cross validate training model"):
            model = xgb.train(param, train, 1000, evals=evallist, early_stopping_rounds=20)
        logger.info("write the model to file--features%d_xgboost%s.model.pickle" % (len(features), str_date))
        pickle.dump(model, open(path_model + "features%d_xgboost%s.model.pickle" % (len(features), str_date), "wb"))
        best_ntree_limit = model.best_ntree_limit

        logger.info("prepare the all train data")
        X_all = XY_all.as_matrix(columns=features)
        Y_all = XY_all.as_matrix(columns=["y"])
        W_all = XY_all.as_matrix(columns=["weights"])
        all_data = xgb.DMatrix(X_all, label=Y_all, feature_names=features, weight=W_all)

        evallist = [(all_data, 'all_data')]
        best_ntree_limit = int(best_ntree_limit * (len(XY_train) + len(XY_validate)) / len(XY_train))
        with Timer("retrain XGBoost with all data"):
            model = xgb.train(param, all_data, best_ntree_limit, evals=evallist)
        logger.info("write the all training data to features%d_xgboost%s.all.model.pickle" % (len(features), str_date))
        pickle.dump(model, open(path_model + "features%d_xgboost%s.all.model.pickle" % (len(features), str_date), "wb"))
    else:
        with Timer("load model from file"):
            model = pickle.load(
                open(path_model + "features%d_xgboost%s.all.model.pickle" % (len(features), str_date), "rb"))

    best_ntree_limit = model.best_ntree_limit

    logger.info("Feature importance:")
    for kv in sorted([(k, v) for k, v in model.get_fscore().items()], key=lambda x: x[1], reverse=True):
        logger.info(kv)

    X_test = test_df.as_matrix(columns=features)
    test = xgb.DMatrix(X_test, feature_names=features)

    return model.predict(test, ntree_limit=best_ntree_limit)


def lightgbm(XY_train, XY_validate, test_df, features, XY_all, str_date, is_file_exist=False):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 24,
        'metric': {'multi_logloss'},
        'is_training_metric': True,
        'max_bin': 255,
        'num_leaves': 64,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 10,
        'min_sum_hessian_in_leaf': 5,
        'num_threads': 8,
    }
    logger.info(params)

    if not is_file_exist:
        train = lgbm.Dataset(XY_train[list(features)], label=XY_train["y"], weight=XY_train["weights"],
                             feature_name=features)
        validate = lgbm.Dataset(XY_validate[list(features)], label=XY_validate["y"], weight=XY_validate["weights"],
                                feature_name=features, reference=train)
        with Timer("LightGBM cross validate training model"):
            model = lgbm.train(params, train, num_boost_round=1000, valid_sets=validate, early_stopping_rounds=20)
        logger.info("write the model to file--features%d_lgbm%s.model.txt" % (len(features), str_date))
        model.save_model(path_model + "features%d_lgbm%s.model.txt" % (len(features), str_date))

        best_iteration = model.best_iteration
        best_iteration = int(best_iteration * len(XY_all) / len(XY_train))
        all_train = lgbm.Dataset(XY_all[list(features)], label=XY_all["y"], weight=XY_all["weights"],
                                 feature_name=features)
        with Timer("retrain lightgbm_lib with all data"):
            model = lgbm.train(params, all_train, num_boost_round=best_iteration)
        logger.info("write the all training data to features%d_lgbm%s.all.model.txt file" % (len(features), str_date))
        model.save_model(path_model + "features%d_lgbm%s.all.model.txt" % (len(features), str_date))
    else:
        with Timer("restore lightgbm_lib model"):
            model = lgbm.Booster(model_file=path_model + "features%d_lgbm%s.all.model.txt" % (len(features), str_date))

    best_iteration = model.best_iteration

    logger.info("Feature importance by split:")
    for kv in sorted([(k, v) for k, v in zip(features, model.feature_importance("split"))], key=lambda x: x[1],
                     reverse=True):
        logger.info(kv)
    logger.info("Feature importance by gain:")
    for kv in sorted([(k, v) for k, v in zip(features, model.feature_importance("gain"))], key=lambda x: x[1],
                     reverse=True):
        logger.info(kv)

    return model.predict(test_df[list(features)], num_iteration=best_iteration)
