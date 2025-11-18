##############################################################################
# STANDARD HYBRID PARTITIONS FOR MULTI-LABEL CLASSIFICATION                  #
# Copyright (C) 2025                                                         #
#                                                                            #
# This code is free software: you can redistribute it and/or modify it under #
# the terms of the GNU General Public License as published by the Free       #
# Software Foundation, either version 3 of the License, or (at your option)  #
# any later version. This code is distributed in the hope that it will be    #
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of     #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General   #
# Public License for more details.                                           #
#                                                                            #
# 1 - Prof Elaine Cecilia Gatto                                              #
# 2 - Prof PhD Ricardo Cerri                                                 #
# 3 - Prof PhD Mauri Ferrandin                                               #
# 4 - Prof PhD Celine Vens                                                   #
# 5 - PhD Felipe Nakano Kenji                                                #
# 6 - Prof PhD Jesse Read                                                    #
#                                                                            #
# 1 = Federal University of São Carlos - UFSCar - https://www2.ufscar.br     #
# Campus São Carlos | Computer Department - DC - https://site.dc.ufscar.br | #
# Post Graduate Program in Computer Science - PPGCC                          # 
# http://ppgcc.dc.ufscar.br | Bioinformatics and Machine Learning Group      #
# BIOMAL - http://www.biomal.ufscar.br                                       # 
#                                                                            # 
# 1 = Federal University of Lavras - UFLA                                    #
#                                                                            # 
# 2 = State University of São Paulo - USP                                    #
#                                                                            # 
# 3 - Federal University of Santa Catarina Campus Blumenau - UFSC            #
# https://ufsc.br/                                                           #
#                                                                            #
# 4 and 5 - Katholieke Universiteit Leuven Campus Kulak Kortrijk Belgium     #
# Medicine Department - https://kulak.kuleuven.be/                           #
# https://kulak.kuleuven.be/nl/over_kulak/faculteiten/geneeskunde            #
#                                                                            #
# 6 - Ecole Polytechnique | Institut Polytechnique de Paris | 1 rue Honoré   #
# d’Estienne d’Orves - 91120 - Palaiseau - FRANCE                            #
#                                                                            #
##############################################################################


import sys
import io
import platform
import os
import time
import pickle
import pandas as pd
import numpy as np

import importlib
import evaluation as eval
importlib.reload(eval)
import measures as ms
importlib.reload(ms)

from collections import Counter
from sklearn.utils.multiclass import type_of_target

import warnings
from sklearn.exceptions import UndefinedMetricWarning

from ml_auprc_roc import *
import ml_auprc_roc as ml
importlib.reload(ml)



if __name__ == '__main__':   


    #true = pd.read_csv("/tmp/d-GnegativeGO/Tested/Split-1/y_true.csv")
    #proba = pd.read_csv("/tmp/d-GnegativeGO/Tested/Split-1/y_pred_proba.csv")
    #directory = "/tmp/d-GnegativeGO/Tested/Split-1"

    # obtendo argumentos da linha de comando    
    true = pd.read_csv(sys.argv[1]) # conjunto de validação    
    proba = pd.read_csv(sys.argv[2]) # conjunto de treino
    directory = sys.argv[3]          # diretório para salvar as predições 
    
    proba = proba.astype(float)
    true = true.astype(int)
    proba = pd.DataFrame(proba.values, columns=true.columns)

    print("\n\n%==============================================%")    
    #print("proba: ", sys.argv[1])
    #print("true: ", sys.argv[2])
    print("directory: ", sys.argv[3])        
    print("%==============================================%\n\n")

    
    # =========== SAVE MEASURES ===========   
    #metrics_df, ignored_df = eval.multilabel_curve_metrics(true, proba)        
    #name = (directory + "/results-python-2.csv") 
    #metrics_df.to_csv(name, index=False)      
    #name = (directory + "/ignored-classes.csv") 
    #ignored_df.to_csv(name, index=False)      
    #print("\nSECOND COMPUTATION!")
    #metrics_df = eval.multilabel_curves_measures(true, pd.DataFrame(proba, columns=true.columns))
    #metrics_df.to_csv(os.path.join(directory, "results-python.csv"), index=False)           

    # =========== SAVE MEASURES ===========   
    fpr_macro, tpr_macro, macro_auc, macro_auc_interp, macro_df = ml.robust_macro_roc_auc(true, proba, verbose=False)
    fpr_micro, tpr_micro, auc_micro = ml.robust_micro_roc_auc(true, proba, verbose=False)
    fpr_w, tpr_w, auc_weighted, auc_df = ml.robust_weighted_roc_auc(true, proba, verbose=False)
    sample_auc_df, samples_auc_mean = ml.robust_sample_roc_auc(true, proba, verbose=False)

    result_df = pd.DataFrame({
        "metric": ["macro_auc", "macro_auc_interp", "micro_auc", "weighted_auc", "samples_auc"],
        "value": [macro_auc, macro_auc_interp, auc_micro, auc_weighted, samples_auc_mean]
    })

    result_df.to_csv(os.path.join(directory, "results-python.csv"), index=False)

    roc_curves_df = pd.DataFrame({
        "Macro_FPR": [fpr_macro],
        "Macro_TPR": [tpr_macro],
        "Micro_FPR": [fpr_micro],
        "Micro_TPR": [tpr_micro],
        "Weighted_FPR": [fpr_w],
        "Weighted_TPR": [tpr_w]
    })

    roc_curves_df.to_pickle(os.path.join(directory, "roc-points.pkl"))
    # df = pd.read_pickle("ROC_curves.pkl")




   
 
