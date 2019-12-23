# -*- coding: utf-8 -*-

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
               'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

all_feature_cols = ['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo',
                   'age', 'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes',
                   'tiprel_1mes', 'indresi', 'indext', 'canal_entrada', 'indfall',
                   'nomprov', 'ind_actividad_cliente', 'renta', 'segmento', 'month']

feature_cols = ['fecha_dato', 'ind_empleado', 'pais_residencia', 'sexo', 'indrel', 'tiprel_1mes',
               'indresi', 'indext', 'canal_entrada', 'indfall', 'nomprov', 'segmento']

map_d = {
    'fecha_dato': {'2015-01-28': 1, '2015-02-28': 2, '2015-03-28': 3, '2015-04-28': 4, '2015-05-28': 5, '2015-06-28': 6,
                   '2015-07-28': 7, '2015-08-28': 8, '2015-09-28': 9, '2015-10-28': 10, '2015-11-28': 11,
                   '2015-12-28': 12, '2016-01-28': 13, '2016-02-28': 14, '2016-03-28': 15, '2016-04-28': 16,
                   '2016-05-28': 17}, 'ind_empleado': {'S': 1, 'A': 2, 'F': 3, 'B': 4, 'UNKNOWN': 5, 'N': 6},
    'pais_residencia': {'MT': 1, 'BM': 2, 'JM': 3, 'ZW': 4, 'DJ': 5, 'KH': 6, 'LV': 7, 'GH': 8, 'ML': 9, 'GI': 10,
                        'KW': 11, 'AL': 12, 'IS': 13, 'KZ': 14, 'SL': 15, 'MM': 16, 'TN': 17, 'TG': 18, 'CF': 19,
                        'LB': 20, 'CD': 21, 'GE': 22, 'LY': 23, 'GM': 24, 'BZ': 25, 'OM': 26, 'VN': 27, 'TW': 28,
                        'BA': 29, 'GW': 30, 'CG': 31, 'MZ': 32, 'PH': 33, 'ET': 34, 'RS': 35, 'HU': 36, 'LT': 37,
                        'EE': 38, 'NZ': 39, 'GA': 40, 'MK': 41, 'MR': 42, 'HK': 43, 'CI': 44, 'GN': 45, 'QA': 46,
                        'NI': 47, 'TR': 48, 'HR': 49, 'EG': 50, 'AO': 51, 'SN': 52, 'KE': 53, 'PA': 54, 'SA': 55,
                        'SK': 56, 'CM': 57, 'PK': 58, 'DZ': 59, 'MD': 60, 'KR': 61, 'PR': 62, 'TH': 63, 'SV': 64,
                        'BY': 65, 'CZ': 66, 'AD': 67, 'SG': 68, 'ZA': 69, 'GQ': 70, 'LU': 71, 'GT': 72, 'NO': 73,
                        'CR': 74, 'IN': 75, 'NG': 76, 'AE': 77, 'DK': 78, 'JP': 79, 'GR': 80, 'HN': 81, 'FI': 82,
                        'MA': 83, 'IE': 84, 'IL': 85, 'DO': 86, 'AU': 87, 'CA': 88, 'BG': 89, 'AT': 90, 'UA': 91,
                        'UY': 92, 'CN': 93, 'PL': 94, 'SE': 95, 'NL': 96, 'CU': 97, 'RU': 98, 'PE': 99, 'CL': 100,
                        'PT': 101, 'PY': 102, 'BO': 103, 'BE': 104, 'CH': 105, 'EC': 106, 'VE': 107, 'BR': 108,
                        'MX': 109, 'RO': 110, 'IT': 111, 'CO': 112, 'US': 113, 'GB': 114, 'DE': 115, 'AR': 116,
                        'FR': 117, 'UNKNOWN': 118, 'ES': 119}, 'sexo': {'UNKNOWN': 1, 'H': 2, 'V': 3},
    'indrel': {99: 1, 1: 2}, 'tiprel_1mes': {'N': 1, 'R': 2, 'P': 3, 'UNKNOWN': 4, 'A': 5, 'I': 6},
    'indresi': {'UNKNOWN': 1, 'N': 2, 'S': 3}, 'indext': {'UNKNOWN': 1, 'S': 2, 'N': 3},
    'canal_entrada': {'KHR': 1, 'KHS': 2, '025': 3, 'KDL': 4, 'KDB': 5, 'KGN': 6, 'KDI': 7, 'KGC': 8, 'KGU': 9,
                      'KHA': 10, 'KBN': 11, 'KFV': 12, 'KCX': 13, 'KEM': 14, 'KBP': 15, 'KBX': 16, 'KCT': 17, 'KAV': 18,
                      'KFB': 19, 'KEQ': 20, 'KBE': 21, 'K00': 22, 'KCP': 23, 'KEE': 24, 'KCO': 25, 'KDH': 26, 'KCR': 27,
                      'KDN': 28, 'KCQ': 29, '004': 30, 'KCJ': 31, 'KCV': 32, 'KEC': 33, 'KCS': 34, 'KBD': 35, 'KAU': 36,
                      'KFE': 37, 'KEU': 38, 'KCE': 39, 'KEF': 40, 'KEB': 41, 'KFR': 42, 'KFM': 43, 'KDA': 44, 'KCF': 45,
                      'KDV': 46, 'KDG': 47, 'KBJ': 48, 'KDZ': 49, 'KDD': 50, 'KEK': 51, 'KBM': 52, 'KBL': 53, 'KBY': 54,
                      'KHP': 55, 'KBS': 56, 'KDF': 57, 'KDW': 58, 'KDE': 59, 'KAK': 60, 'KEV': 61, 'KEO': 62, 'KEA': 63,
                      'KFI': 64, 'KBV': 65, 'KEI': 66, 'KCK': 67, 'KGW': 68, 'KDP': 69, 'KCU': 70, 'KDQ': 71, 'KCN': 72,
                      'KBW': 73, 'KBB': 74, 'KDT': 75, 'KAN': 76, 'KCA': 77, 'KDC': 78, 'KEH': 79, 'KBG': 80, 'KDX': 81,
                      'KDO': 82, 'KBR': 83, 'KEG': 84, 'KDS': 85, 'KDY': 86, 'KEZ': 87, 'KDM': 88, 'KEL': 89, 'KFH': 90,
                      'KDU': 91, 'KED': 92, 'KBU': 93, 'KCM': 94, 'KCD': 95, 'KBF': 96, 'KFL': 97, 'KFK': 98, 'KGY': 99,
                      'KBQ': 100, 'KCL': 101, 'KFN': 102, 'KFU': 103, 'KEN': 104, 'KCB': 105, 'KCG': 106, 'KFF': 107,
                      'KEW': 108, 'KES': 109, 'KFJ': 110, 'KAO': 111, 'KFS': 112, 'KFG': 113, 'KBH': 114, 'KBO': 115,
                      'KAL': 116, 'KAC': 117, 'KFT': 118, 'KDR': 119, 'KGV': 120, 'KHO': 121, 'KEJ': 122, 'KGX': 123,
                      'KFP': 124, 'KAD': 125, 'KAM': 126, 'KAP': 127, 'KHC': 128, 'KAQ': 129, 'KHF': 130, 'KCH': 131,
                      'KAJ': 132, 'KAH': 133, 'KCI': 134, '013': 135, '007': 136, 'KAF': 137, 'KAZ': 138, 'KAR': 139,
                      'KAW': 140, 'KEY': 141, 'KAI': 142, 'KFD': 143, 'KHL': 144, 'KBZ': 145, 'KCC': 146, 'KAE': 147,
                      'KAB': 148, 'KAA': 149, 'KAY': 150, 'KAG': 151, 'RED': 152, 'KAS': 153, 'KHN': 154, 'KHD': 155,
                      'KHM': 156, 'UNKNOWN': 157, 'KHK': 158, 'KFA': 159, 'KHQ': 160, 'KFC': 161, 'KAT': 162,
                      'KHE': 163}, 'indfall': {'UNKNOWN': 1, 'S': 2, 'N': 3},
    'nomprov': {'CEUTA': 1, 'MELILLA': 2, 'SORIA': 3, 'TERUEL': 4, 'ALAVA': 5, 'AVILA': 6, 'HUESCA': 7, 'SEGOVIA': 8,
                'PALENCIA': 9, 'ZAMORA': 10, 'CUENCA': 11, 'ALMERIA': 12, 'JAEN': 13, 'GUADALAJARA': 14,
                'SANTA CRUZ DE TENERIFE': 15, 'GIPUZKOA': 16, 'LERIDA': 17, 'LEON': 18, 'OURENSE': 19, 'LUGO': 20,
                'RIOJA, LA': 21, 'NAVARRA': 22, 'GIRONA': 23, 'UNKNOWN': 24, 'BURGOS': 25, 'TARRAGONA': 26,
                'CASTELLON': 27, 'ALBACETE': 28, 'CIUDAD REAL': 29, 'HUELVA': 30, 'BALEARS, ILLES': 31, 'CACERES': 32,
                'CORDOBA': 33, 'CANTABRIA': 34, 'SALAMANCA': 35, 'GRANADA': 36, 'TOLEDO': 37, 'BIZKAIA': 38,
                'BADAJOZ': 39, 'PALMAS, LAS': 40, 'VALLADOLID': 41, 'ASTURIAS': 42, 'PONTEVEDRA': 43, 'CADIZ': 44,
                'ALICANTE': 45, 'ZARAGOZA': 46, 'MALAGA': 47, 'MURCIA': 48, 'CORUNA, A': 49, 'SEVILLA': 50,
                'VALENCIA': 51, 'BARCELONA': 52, 'MADRID': 53},
    'segmento': {'UNKNOWN': 1, '01 - TOP': 2, '03 - UNIVERSITARIO': 3, '02 - PARTICULARES': 4}}