import time
import logging
import numpy as np

path_input = "/home/ubuntu/filesystem/"
path_result = "/home/ubuntu/filesystem/Result/"
path_model = "/home/ubuntu/filesystem/Model/"
path_log = "/home/ubuntu/filesystem/Logging/"

# the following configure is only write to file, we also want to print logging in console
# logging.basicConfig(level=logging.INFO, filename=path_log + "main.log", filemode='w',
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler(path_log + 'main.log', mode='w')
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

products = ('ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
            'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
            'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
            'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
            'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1')


class Timer:
    def __init__(self, text=None):
        self.text = text

    def __enter__(self):
        self.cup = time.clock()
        self.time = time.time()
        if self.text:
            logger.debug("{}...".format(self.text))
        return self

    def __exit__(self, *args):
        self.cup = time.clock() - self.cup
        self.time = time.time() - self.time
        if self.text:
            logger.info('%s cost cup : %0.3f' % (self.text, self.cup))
            logger.info('%s cost time : %0.3f' % (self.text, self.time))


def date_to_int1(date_str):
    y, m, d = [int(a) for a in date_str.strip().split('-')]
    int_date = (y - 2015) * 12 + m
    return int_date


def date_to_int2(date_str):
    y, m, d = [int(a) for a in date_str.strip().split('-')]
    int_date = y * 12 + m
    return int_date


def apk(actual, predicted, k=7, default=0.0):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hit = 0.0

    if not actual:
        return default

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hit += 1
            score += num_hit / (i + 1.0)

    return score / min(k, len(actual))


def mapk(actual, predicted, k=7, default=0.0):
    return np.mean([apk(a, p, k, default) for (a, p) in zip(actual, predicted)])
