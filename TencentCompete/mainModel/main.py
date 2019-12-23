from mainModel.feature import *
from mainModel.engine import *


def load_data(iscode=True):
    if iscode:
        with open(path_origin + origin_all_code, "rb") as fr:
            allX = pickle.load(fr)
    else:
        allX = pd.read_csv(path_origin + origin_all)
        allX = allX.values
    with open(path_origin + label_file, "rb") as fr:
        label = pickle.load(fr)
    return allX[:len(label)], allX[len(label):], label


def load_data_simple(iscode=True):
    if iscode:
        with open(path_simple + simple_all_code, "rb") as fr:
            allX = pickle.load(fr)
    else:
        allX = pd.read_csv(path_simple + simple_all)
        allX = allX[feat]
        allX = allX.values
    with open(path_simple + label_file, "rb") as fr:
        label = pickle.load(fr)
    return allX[:len(label)], allX[len(label):], label


def train_model():
    # Xtrain, Xtest, label = load_data(iscode=False)
    Xtrain, Xtest, label = load_data_simple(iscode=True)
    # logistic_regression(Xtrain, Xtest, label)
    # random_forest(Xtrain, Xtest, label)
    # gbdt(Xtrain, Xtest, label)
    # base_model_kfold(Xtrain, Xtest, label)
    # base_model(Xtrain, Xtest, label)
    # base_model_grid_search(Xtrain, Xtest, label)
    # xgboost(Xtrain, Xtest, label)
    xgboost1(Xtrain, Xtest, label)
    # xgboost2(Xtrain, Xtest, label)


if __name__ == "__main__":
    with Timer("\n\nthe all program ..."):
        # origin()
        # simple()
        train_model()


