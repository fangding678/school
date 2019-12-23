import scipy as sp
import logging
import time

# path_pre = "D:/Codes/Data/TecentCompete/pre/"
# path_origin = "D:/Codes/Data/TecentCompete/origin/"
path_pre = "/home/ubuntu/tecent/pre/"
path_origin = "/home/ubuntu/cfilesystem/tecent/origin/"
path_simple = "/home/ubuntu/cfilesystem/tecent/simple/"
path_log = "/home/ubuntu/tecent/Log/"
path_result = "/home/ubuntu/tecent/result/"
path_model = "/home/ubuntu/tecent/model/"
path_temporary = "/home/ubuntu/cfilesystem/tecent/temporary/"

pre_file_ad = "ad.csv"
pre_file_app_categories = "app_categories.csv"
pre_file_position = "position.csv"
pre_file_train = "train.csv"
pre_file_test = "test.csv"
pre_file_user = "user.csv"
pre_file_user_app_actions = "user_app_actions.csv"
pre_file_user_installedapps = "user_installedapps.csv"
pre_file_install_dict = "install_dict.pickle"

origin_file_train = "origin_train.csv"
origin_file_test = "origin_test.csv"
origin_all = "allX.csv"
origin_all_code = "allX_code.pickle"
label_file = "label.pickle"

simple_all = "all_simple.csv"
simple_all_code = "all_simple_code.pickle"

temp_file_installed = "installed.pickle"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler(path_log + 'main_1.log', mode='a')
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

feat = ['clickTime_day', 'clickTime_week', 'clickTime_hour', 'creativeID', 'positionID', 'connectionType',
        'telecomsOperator', 'sitesetID', 'positionType', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform',
        'appCategory', 'appCategory_1', 'appCategory_2', 'age_label', 'gender', 'education', 'marriageStatus',
        'haveBaby', 'hometown', 'residence', 'hometown_nomprov', 'residence_nomprov', 'same_city', 'same_prov',
        'appisin']


class Timer:
    def __init__(self, text=None):
        self.text = text

    def __enter__(self):
        self.cpu = time.clock()
        self.time = time.time()
        if self.text:
            logger.info("{}...".format(self.text))
        return self

    def __exit__(self, *args):
        self.cpu = time.clock() - self.cpu
        self.time = time.time() - self.time
        if self.text:
            logger.info("%s cost cpu : %0.3fs" % (self.text, self.cpu))
            logger.info("%s cost time : %0.3fs" % (self.text, self.time))


def logloss(act, pred, epsilon=1e-15):
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


def age_label(age):
    if age > 50:
        age = 7
    elif age > 40:
        age = 6
    elif age > 30:
        age = 5
    elif age > 22:
        age = 4
    elif age > 16:
        age = 3
    elif age > 10:
        age = 2
    elif age > 0:
        age = 1
    else:
        age = 0
    return age


def f1(x):
    if x["hometown"] == 0 and x["residence"] == 0:
        return 0
    elif x["hometown"] == 0 or x["residence"] == 0:
        return 1
    elif x["hometown"] != x["residence"]:
        return 2
    else:
        return 3


def f2(x):
    if x["hometown_nomprov"] == 0 and x["residence_nomprov"] == 0:
        return 0
    elif x["hometown_nomprov"] == 0 or x["residence_nomprov"] == 0:
        return 1
    elif x["hometown_nomprov"] != x["residence_nomprov"]:
        return 2
    else:
        return 3
