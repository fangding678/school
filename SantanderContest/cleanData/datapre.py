import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

input_path = 'D:/Codes/Data/SantanderProductRecommendation/'
output_path = 'D:/Codes/Data/SantanderProductRecommendation/ResultData/'


def datapre1():
    print("Reading files and getting top products..")
    train = pd.read_csv(input_path + "train_ver2.csv", dtype='float16',
                        usecols=['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                                 'ind_cder_fin_ult1',
                                 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                                 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                                 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                                 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                                 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'])
    count_dict = {}
    for col_name in list(train.columns):
        if col_name != 'ncodpers':
            count_dict[col_name] = np.sum(train[col_name].astype('float64'))

    top_products = sorted(count_dict, key=count_dict.get, reverse=True)

    print("Drop duplicates and keep last one")
    train['ncodpers'] = train['ncodpers'].astype('int')
    train = train.drop_duplicates('ncodpers', keep='last')

    print("Read sample submission and merge with train..")
    sub = pd.read_csv(input_path + "sample_submission.csv")
    sub = sub.merge(train, on='ncodpers', how='left')
    del train
    sub.fillna(0, inplace=True)

    print("Get the top products which are not already bought..")
    ofile = open("simple_btb_v2.0.csv", "w")
    writer = csv.writer(ofile)
    writer.writerow(['ncodpers', 'added_products'])
    for ind, row in sub.iterrows():
        cust_id = row['ncodpers']
        top7_products = []
        for product in top_products:
            if int(row[product]) == 0:
                top7_products.append(str(product))
                if len(top7_products) == 7:
                    break
        writer.writerow([cust_id, " ".join(top7_products)])
    ofile.close()


# 为所有用户推荐最热门的7种业务
def trail():
    filename = "D:/Codes/SantanderContest/Result/sample_submission.csv"
    filename1 = "D:/Codes/SantanderContest/Result/submission1.csv"
    d = {'ind_cno_fin_ult1': 1103620.0, 'ind_nomina_ult1': 745961.0, 'ind_pres_fin_ult1': 35857.0,
         'ind_cco_fin_ult1': 8945588.0, 'ind_deco_fin_ult1': 24275.0, 'ind_dela_fin_ult1': 586381.0,
         'ind_tjcr_fin_ult1': 605786.0, 'ind_nom_pens_ult1': 810085.0, 'ind_deme_fin_ult1': 22668.0,
         'ind_ctma_fin_ult1': 132742.0, 'ind_hip_fin_ult1': 80336.0, 'ind_valo_fin_ult1': 349475.0,
         'ind_aval_fin_ult1': 316.0, 'ind_ctop_fin_ult1': 1760616.0, 'ind_ecue_fin_ult1': 1129227.0,
         'ind_cder_fin_ult1': 5376.0, 'ind_viv_fin_ult1': 52511.0, 'ind_ctpp_fin_ult1': 591008.0,
         'ind_recibo_ult1': 1745712.0, 'ind_fond_fin_ult1': 252284.0, 'ind_plan_fin_ult1': 125159.0,
         'ind_reca_fin_ult1': 716980.0, 'ind_ahor_fin_ult1': 1396.0, 'ind_ctju_fin_ult1': 129297.0}
    dd = sorted(d.items(), reverse=True, key=lambda d: d[1])[:7]
    s = ""
    for i in dd:
        s += i[0] + " "
    s = s[0:-1]
    print(s)
    ss = ""
    fr = open(filename)
    fw = open(filename1, "w")
    for line in fr.readlines():
        ss = line.split(",")[0]
        ss += "," + s + "\n"
        fw.write(ss)


# 为所有用户推荐上一个月的业务，也就是说业务不变
def trail2():
    filetest = "valid_test.csv"
    filesample = "Result/sample_submission.csv"
    filesubmission2 = "Result/submission2.csv"
    '''
    vtrain = pd.read_csv(input_path+"valid_test.csv", dtype='int',
                         usecols=['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                                  'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                                  'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                                  'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                                  'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                                  'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'])
    #vvdf = pd.read_csv(input_path+"valid_test.csv")
    # print(vvdf)
    #vtrain['ncodpers'] = vtrain['ncodpers'].astype('int')
    print(len(vtrain))
    print(vtrain)
    '''
    fr = open(input_path + filetest, 'r')
    fw = open(input_path + filesubmission2, 'w')
    l = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
         'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
         'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
         'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
         'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
         'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
    d = {}
    ll = ['ind_cco_fin_ult1', 'ind_ctop_fin_ult1', 'ind_recibo_ult1', 'ind_ecue_fin_ult1',
          'ind_cno_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']
    for line in fr.readline():
        line = line.strip().split(',')
        k = line[1]
        d[k] = []
        i = 0
        for l in line[24:]:
            if l == '1':
                d[k].append(l[i])
            i += 1
        if len(d[k]) == 0:
            d[k] = ll
    for k, v in d.items().sort():
        ss = ''
        ss += str(k)
        for i in v:
            ss += ' ' + i
        ss += '\n'
        fw.write()


def example():
    train = pd.read_csv(input_path + "train_ver2.csv", dtype='float16',
                        usecols=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                                 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1',
                                 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1',
                                 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                                 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1',
                                 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1',
                                 'ind_recibo_ult1'])
    count_dict = {}
    for col_name in list(train.columns):
        count_dict[col_name] = np.sum(train[col_name].astype('float64'))
    print(count_dict)
    sorted(count_dict.items(), key=lambda d: d[1])
    print(count_dict)


def Splitdata():
    train_origin = input_path + "train_ver2.csv"
    train_split = input_path + "valid_train.csv"
    test_split = input_path + "valid_test.csv"
    lcol = ["fecha_dato", "ncodpers", "ind_empleado", "pais_residencia",
            "sexo", "age", "fecha_alta", "ind_nuevo", "antiguedad",
            "indrel", "ult_fec_cli_1t", "indrel_1mes", "tiprel_1mes",
            "indresi", "indext", "conyuemp", "canal_entrada", "indfall",
            "tipodom", "cod_prov", "nomprov", "ind_actividad_cliente",
            "renta", "segmento", "ind_ahor_fin_ult1", "ind_aval_fin_ult1",
            "ind_cco_fin_ult1", "ind_cder_fin_ult1", "ind_cno_fin_ult1",
            "ind_ctju_fin_ult1", "ind_ctma_fin_ult1", "ind_ctop_fin_ult1",
            "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1",
            "ind_dela_fin_ult1", "ind_ecue_fin_ult1", "ind_fond_fin_ult1",
            "ind_hip_fin_ult1", "ind_plan_fin_ult1", "ind_pres_fin_ult1",
            "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
            "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1",
            "ind_recibo_ult1"]
    fr = open(train_origin)
    fwtr = open(train_split, "w")
    fwte = open(test_split, "w")
    fwte.write(str(lcol)[1:-1] + '\n')
    for l in fr.readlines():
        line = l.split(',')
        if line[0] != "2016-05-28":
            fwtr.writelines(l)
        else:
            fwte.writelines(l)


if __name__ == "__main__":
    st = time.time()
    # Splitdata()
    # trail()
    trail2()
    # example()
    # datapre1()
    et = time.time()
    print(str(et - st) + 's')
