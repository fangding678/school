import pandas as pd
from mainModel.utils import *
from sklearn.preprocessing import OneHotEncoder
import pickle
import numpy as np

origin_features = ['clickTime', 'creativeID', 'userID', 'positionID', 'connectionType', 'telecomsOperator', 'sitesetID',
                   'positionType', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory', 'age',
                   'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence']


def origin(exist=True):
    if not exist:
        logger.info("create the origin file data")
        ad = pd.read_csv(path_pre + pre_file_ad)
        app_categories = pd.read_csv(path_pre + pre_file_app_categories)
        origin_ad = pd.merge(ad, app_categories, on="appID", how="left")
        position = pd.read_csv(path_pre + pre_file_position)
        user = pd.read_csv(path_pre + pre_file_user)
        train = pd.read_csv(path_pre + pre_file_train)
        test = pd.read_csv(path_pre + pre_file_test)
        train = pd.merge(train, position, on="positionID", how="left")
        train = pd.merge(train, origin_ad, on="creativeID", how="left")
        train = pd.merge(train, user, on="userID", how="left")
        test = pd.merge(test, position, on="positionID", how="left")
        test = pd.merge(test, origin_ad, on="creativeID", how="left")
        test = pd.merge(test, user, on="userID", how="left")
        train.to_csv(path_origin + origin_file_train, index=False)
        test.to_csv(path_origin + origin_file_test, index=False)
    else:
        logger.info("read the origin data from file")
        train = pd.read_csv(path_origin + origin_file_train)
        test = pd.read_csv(path_origin + origin_file_test)
    origin_features1 = origin_features[:2] + origin_features[3:]
    label = train["label"].values
    all_df = pd.concat([train[origin_features1], test[origin_features1]])
    with open(path_origin + origin_all, "wb") as fw:
        pickle.dump(all_df, fw)
    enc = OneHotEncoder()
    logger.info("OneHot Coding...")
    all_df = enc.fit_transform(all_df)
    logger.info("write origin feature data to file")
    with open(path_origin + origin_all_code, "wb") as fw:
        pickle.dump(all_df, fw)
    with open(path_origin + label_file, "wb") as fw:
        pickle.dump(label, fw)


def simple():
    train = pd.read_csv(path_origin + origin_file_train)
    test = pd.read_csv(path_origin + origin_file_test)
    df_all = pd.concat([train[origin_features], test[origin_features]])
    user_installedapps = pd.read_csv(path_pre + pre_file_user_installedapps)
    d = user_installedapps.groupby('userID')['appID'].apply(lambda x: set(x))
    with open(path_temporary + temp_file_installed, "wb") as fw:
        pickle.dump(d, fw)
    features = []
    # df_all["clickTime_day"] = df_all["clickTime"] // 10000
    df_all["clickTime_week"] = (df_all["clickTime"] // 10000) % 7
    df_all["clickTime_hour"] = (df_all["clickTime"] // 100) % 100
    # features.append("clickTime_day")
    features.append("clickTime_week")
    features.append("clickTime_hour")
    features += ['creativeID', 'userID', 'positionID', 'connectionType', 'telecomsOperator', 'sitesetID',
                 'positionType', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', "appCategory"]
    df_all["appCategory"] = df_all["appCategory"].apply(lambda x: x*100 if x < 10 else x)
    df_all["appCategory_1"] = df_all["appCategory"] // 100
    df_all["appCategory_2"] = df_all["appCategory"] % 100
    features.append("appCategory_1")
    features.append("appCategory_2")
    df_all["age_label"] = df_all["age"].apply(age_label)
    features.append("age_label")
    features += ['gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence']
    df_all["hometown_nomprov"] = df_all["hometown"] // 100
    df_all["residence_nomprov"] = df_all["residence"] // 100
    features.append("hometown_nomprov")
    features.append("residence_nomprov")
    df_all["same_city"] = df_all.apply(f1, axis=1)
    df_all["same_prov"] = df_all.apply(f2, axis=1)
    features.append("same_city")
    features.append("same_prov")
    df_all["appisin"] = df_all.apply(lambda x: 1 if x["appID"] in d.get([x["userID"]], set()) else 0, axis=1)
    features.append("appisin")
    logger.info("the feature is ...")
    logger.info(features)
    df_all = df_all[features]
    df_all.to_csv(path_simple + simple_all, index=False)

    enc = OneHotEncoder()
    logger.info("OneHot Coding...")
    df_all = enc.fit_transform(df_all[feat])
    logger.info("write the OneHot data to file...")
    with open(path_simple + simple_all_code, "wb") as fw:
        pickle.dump(df_all, fw)


def complicate():
    pass


def model_construct():
    pass
