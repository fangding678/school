from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
from imblearn import combine
from mainModel.utils import *
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, vstack


def base_model(Xtrain, Xtest, label):
    classifers = {
        "logisticregression": LogisticRegression(verbose=1, n_jobs=4)
        # "randomforest": RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=4),
        # "gbdt": GradientBoostingClassifier(verbose=1)
    }
    logger.info("running the base model")
    for name, clf in classifers.items():
        logger.info("fit the %s model" % name)
        clf.fit(Xtrain, label)
        pred = clf.predict_proba(Xtest)[:, 1]
        id = np.arange(1, len(pred)+1)
        res = pd.DataFrame()
        res["instanceID"] = id
        res["prob"] = pred
        res.to_csv(path_result + "submission_simple_code1_%s.csv" % name, index=False)


def base_model_kfold(Xtrain, Xtest, label):
    classifers = {
        "logisticregression": LogisticRegression(verbose=1, n_jobs=2),
        "randomforest": RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=2),
        "gbdt": GradientBoostingClassifier(verbose=1)
    }
    kf = KFold(n_splits=3)
    logger.info("running the base model kfold")
    for name, clf in classifers.items():
        logger.info("model %s trying cross valid..." % name)
        for k, (val_tr_i, val_te_i) in enumerate(kf.split(Xtrain, label)):
            clf.fit(Xtrain[val_tr_i], label[val_tr_i])

            score1 = clf.score(Xtrain[val_tr_i], label[val_tr_i])
            pred1 = clf.predict_proba(Xtrain[val_tr_i])[:, 1]
            loss1 = logloss(act=label[val_tr_i], pred=pred1)
            logger.info("[valid fold {0}], score: {1:.8f}".format(k, score1))
            logger.info("[valid fold {0}], logloss: {1:.8f}\n".format(k, loss1))

            score2 = clf.score(Xtrain[val_te_i], label[val_te_i])
            pred2 = clf.predict_proba(Xtrain[val_te_i])[:, 1]
            loss2 = logloss(act=label[val_te_i], pred=pred2)
            logger.info("[fold {0}], score: {1:.8f}".format(k, score2))
            logger.info("[fold {0}], logloss: {1:.8f}\n".format(k, loss2))


def base_model_grid_search(Xtrain, Xtest, label):
    classifers = {
        # "logisticregression": LogisticRegression(verbose=1, n_jobs=4),
        "randomforest": RandomForestClassifier(verbose=1, n_jobs=4),
        "gbdt": GradientBoostingClassifier(verbose=1)
    }
    param_dict = {
        # "logisticregression": {},
        "randomforest": {"n_estimators": [10, 20, 30, 50]},
        "gbdt": {"n_estimators": [100, 200, 300, 500], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]}
    }
    for name, clf in classifers.items():
        grad_search = GridSearchCV(clf, cv=3, param_grid=param_dict[name], verbose=1)
        grad_search_clf = grad_search.fit(Xtrain, label)
        pred = grad_search_clf.predict_proba(Xtest)[:, 1]
        logger.info(grad_search.best_estimator_)

        logger.info("write the result to file...")
        id = np.arange(1, len(pred) + 1)
        res = pd.DataFrame()
        res["instanceID"] = id
        res["prob"] = pred
        res.to_csv(path_result + "submission_simple_grid1_%s.csv" % name, index=False)


def logistic_regression(Xtrain, Xtest, label):
    clf = LogisticRegression(verbose=2, n_jobs=4)
    kf = KFold(n_splits=3, random_state=True, shuffle=True)
    logger.info("model LR trying cross valid...")
    for k, (valid_train_index, valid_test_index) in enumerate(kf.split(Xtrain, label)):
        clf.fit(Xtrain[valid_train_index], label[valid_train_index])
        score = clf.score(Xtrain[valid_test_index], label[valid_test_index])
        pred = clf.predict_proba(Xtrain[valid_test_index])[:, 1]
        loss = logloss(label[valid_test_index], pred)
        logger.info("[fold {0}], score: {1:.5f}".format(k, score))
        logger.info("[fold {0}], logloss: {1:.5f}".format(k, loss))
        logger.info("\n")


def random_forest(Xtrain, Xtest, label):
    clf = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=-1)
    kf = KFold(n_splits=2, random_state=True, shuffle=True)
    logger.info("model RF trying cross valid...")
    for k, (valid_train_index, valid_test_index) in enumerate(kf.split(Xtrain, label)):
        clf.fit(Xtrain[valid_train_index], label[valid_train_index])
        score = clf.score(Xtrain[valid_test_index], label[valid_test_index])
        pred = clf.predict_proba(Xtrain[valid_test_index])[:, 1]
        loss = logloss(label[valid_test_index], pred)
        logger.info("[fold {0}], score: {1:.5f}".format(k, score))
        logger.info("[fold {0}], logloss: {1:.5f}".format(k, loss))
        logger.info(clf.feature_importances_)
        score1 = clf.score(Xtrain[valid_train_index], label[valid_train_index])
        pred1 = clf.predict_proba(Xtrain[valid_train_index])[:, 1]
        loss1 = logloss(label[valid_train_index], pred1)
        logger.info("[valid fold {0}], score: {1:.5f}".format(k, score1))
        logger.info("[valid fold {0}], logloss: {1:.5f}".format(k, loss1))
        logger.info("\n")


def gbdt(Xtrain, Xtest, label):
    clf = GradientBoostingClassifier(verbose=2)
    kf = KFold(n_splits=3, random_state=True, shuffle=True)
    logger.info("model GBDT trying cross valid...")
    assert len(Xtrain) == len(label)
    for k, (valid_train_index, valid_test_index) in enumerate(kf.split(Xtrain, label)):
        clf.fit(Xtrain[valid_train_index], label[valid_train_index])
        score = clf.score(Xtrain[valid_test_index], label[valid_test_index])
        pred = clf.predict_proba(Xtrain[valid_test_index])[:, 1]
        loss = logloss(label[valid_test_index], pred)
        logger.info("[fold {0}], score: {1:.5f}".format(k, score))
        logger.info("[fold {0}], logloss: {1:.5f}".format(k, loss))
        logger.info(clf.feature_importances_)
        score1 = clf.score(Xtrain[valid_train_index], label[valid_train_index])
        pred1 = clf.predict_proba(Xtrain[valid_train_index])[:, 1]
        loss1 = logloss(label[valid_train_index], pred1)
        logger.info("[valid fold {0}], score: {1:.5f}".format(k, score1))
        logger.info("[valid fold {0}], logloss: {1:.5f}".format(k, loss1))
        logger.info("\n")


def xgboost(Xtrain, Xtest, label):
    param_dict = {
        'objective': ['binary:logistic'],
        # 'eta': [0.05, 0.1, 0.2],
        'min_child_weight': [6, 8, 10],
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 10, 12],
        'silent': [1],
        # 'nthread': [6],
        # 'eval_metric': ['mlogloss'],
        'colsample_bytree': [0.8],
        'colsample_bylevel': [0.9],
        # 'num_class': [2]
    }
    xgb_model = xgb.XGBClassifier()
    clf = GridSearchCV(xgb_model, param_dict, n_jobs=4, verbose=1)
    clf.fit(Xtrain, label)
    logger.info(clf.best_score_)
    logger.info(clf.best_params_)
    with open(path_model + "xgb_simple_best1.pickle", "wb") as fw:
        pickle.dump(clf, fw)
    pred = clf.predict_proba(Xtest)[:, 1]
    id = np.arange(1, len(pred)+1)
    res = pd.DataFrame()
    res["instanceID"] = id
    res["prob"] = pred
    res.to_csv(path_result + "submission_simple_xgb_best1.csv", index=False)


def xgboost1(Xtrain, Xtest, label):
    clf = xgb.XGBClassifier()
    logger.info("fit the xgb model")
    clf.fit(Xtrain, label)
    pred = clf.predict_proba(Xtest)[:, 1]
    id = np.arange(1, len(pred) + 1)
    res = pd.DataFrame()
    res["instanceID"] = id
    res["prob"] = pred
    res.to_csv(path_result + "submission_simple_code_xgb3.csv", index=False)


def xgboost2(Xtrain, Xtest, label):
    param = {'max_depth': 6, 'eta': 1, 'objective': 'binary:logistic'}
    lenX = Xtest.shape[0]
    ll = [0] * lenX
    label1 = np.array(label.tolist() + ll)
    Xall = vstack([Xtrain, Xtest])
    Xall1 = xgb.DMatrix(data=Xall, label=label1)
    Xall2 = xgb.DMatrix(data=Xall)
    Xtrain = Xall1.slice(range(len(label)))
    Xtest = Xall2.slice(range(len(label), Xall2.num_row()))
    # watchlist = [(Xtest, "eval"), (Xtrain, "train")]
    logger.info("fit the xgb model")
    bst = xgb.train(params=param, dtrain=Xtrain)
    # pickle.dump(bst, open(path_model + "xgb_model.pickle", "wb"))
    # pickle.dump(t, open(path_model + "data.pickle", "wb"))
    pred = bst.predict(Xtest)
    id = np.arange(1, len(pred) + 1)
    res = pd.DataFrame()
    res["instanceID"] = id
    res["prob"] = pred
    res.to_csv(path_result + "submission_simple_code_xgb2.csv", index=False)

