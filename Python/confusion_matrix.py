##############################################################################
# LABEL CLUSTERS CHAINS FOR MULTILABEL CLASSIFICATION                        #
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
# Prof. Elaine Cecilia Gatto - UFLA - Lavras, Minas Gerais, Brazil           #
# Prof. Ricardo Cerri - USP - São Carlos, São Paulo, Brazil                  #
# Prof. Mauri Ferrandin - UFSC - Blumenau, Santa Catarina, Brazil            #
# Prof. Celine Vens - Ku Leuven - Kortrijik, West Flanders, Belgium          #
# PhD Felipe Nakano Kenji - Ku Leuven - Kortrijik, West Flanders, Belgium    #
#                                                                            #
# BIOMAL - http://www.biomal.ufscar.br                                       #
#                                                                            #
##############################################################################

########################################################################
#                                                                      #
########################################################################
import sys
import platform
import os

#FolderRoot = os.path.expanduser('/lapix/arquivos/elaine/GlobalPartitions/Python')
#os.chdir(FolderRoot)
#current_directory = os.getcwd()
#sys.path.append('..')

import pandas as pd
import numpy as np


########################################################################
#                                                                      #
########################################################################
def get_all_measures_names():
    """
    Returns a dictionary with hierarchical measure names for multi-label classification.

    The dictionary is organized into categories, each containing a list of measure names.

    Returns:
    -------
    dict
        A dictionary with keys representing categories and values as lists of measure names.

    Example:
    -------
    >>> measures = get_all_measures_names()
    >>> print(measures['macro-based'])
    ['macro-auprc', 'macro-F1', 'macro-precision', 'macro-recall', 'macro-jaccard', 'macro-roc_auc']
    """
    return {
        'all': [
            "bipartition",            
            "label-problem",
            "ranking",
            "scores"
        ],
        'bipartition': [
            "example-based",
            "label-based"            
        ],
        'ranking': [
            "average-precision",
            "coverage",
            "margin-loss"
            "one-error",            
            "ranking-loss"            
            
        ],
        'label-based': [            
            "macro-based",
            "micro-based"            
        ],
        'example-based': [
            "accuracy",
            "hamming-loss",
            "zero-one-loss"
        ],
        'macro-based': [
            "macro-auprc",
            "macro-f1",
            "macro-precision",
            "macro-recall",
            "macro-jaccard",
            "macro-roc-auc"
        ],
        'micro-based': [
            "micro-auprc",
            "micro-f1",
            "micro-jaccard",
            "micro-precision",
            "micro-recall",            
            "micro-roc-auc"
        ],
        'weighted-based': [
            "weighted-auprc",
            "weighted-f1",
            "weighted-jaccard",
            "weighted-precision",
            "weighted-recall",            
            "weighted-roc-auc"
        ],
        'samples-based': [
            "samples-auprc",
            "samples-f1",
            "samples-jaccard"
            "samples-precision",
            "samples-recall"            
        ],       
        'label-problem': [
            "clp",
            "mlp",
            "wlp"
        ],
        'scores': [
            "auprc",
            "roc"
        ]
    }


########################################################################
#                                                                      #
########################################################################
def number_of_instances(dataset):
    """
    Returns the total number of instances (rows) in a dataset.

    Parameters:
    -------
    dataset (pd.DataFrame): The DataFrame for which the number of instances will be counted.

    Returns:
    -------
    int: The total number of instances (rows) in the dataset.

    Example:
    >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> get_number_of_instances(df)
    3
    """
    return len(dataset)



########################################################################
#                                                                      #
########################################################################
def number_of_labels(true_labels):
    """
    Returns the total number of labels (columns) in a dataset.

    Parameters:
    -------
    true_labels (pd.DataFrame): The DataFrame containing the true labels with binary values (0 or 1).

    Returns:
    -------
    int: The total number of labels (columns) in the dataset.

    Example:
    >>> labels = {'Label1': [1, 0, 1], 'Label2': [0, 1, 1]}
    >>> df_labels = pd.DataFrame(labels)
    >>> get_number_of_labels(df_labels)
    2
    """
    return len(true_labels.columns)


########################################################################
#                                                                      #
########################################################################
def positive_instances(dataset):
    """
    Returns the count of positive instances for each label in a dataset.

    Parameters:
    -------
    dataset (pd.DataFrame): The DataFrame where each column represents a label and contains binary values (0 or 1).

    Returns:
    -------
    pd.Series: A Series containing the count of positive instances for each label (i.e., the sum of each column).

    Example:
    >>> data = {'Label1': [1, 0, 1], 'Label2': [0, 1, 1]}
    >>> df = pd.DataFrame(data)
    >>> get_positive_instances(df)
    Label1    2
    Label2    2
    dtype: int64
    """
    return dataset.sum()


########################################################################
#                                                                      #
########################################################################
def negative_instances(dataset, positive_instances):
    """
    Returns the count of negative instances for each label in a dataset.

    Parameters:
    -------
    dataset (pd.DataFrame): The DataFrame where each column represents a label and contains binary values (0 or 1).
    
    positive_instances (pd.Series): A Series containing the count of positive instances for each label.

    Returns:
    -------
    pd.Series: A Series containing the count of negative instances for each label.

    Example:
    >>> data = {'Label1': [1, 0, 1], 'Label2': [0, 1, 1]}
    >>> df = pd.DataFrame(data)
    >>> positive_instances = df.sum()
    >>> get_negative_instances(df, positive_instances)
    Label1    1
    Label2    1
    dtype: int64
    """
    total_instances = len(dataset)
    negative_instances = total_instances - positive_instances
    return negative_instances


########################################################################
#                                                                      #
########################################################################
def mlem_true_1(true_labels):
    """
    Returns a DataFrame indicating where the true labels are equal to 1.

    Parameters:
    -------
    true_labels (pd.DataFrame): The DataFrame containing the true labels with binary values (0 or 1).

    Returns:
    -------
    pd.DataFrame: A DataFrame of the same shape as `true_labels`, where cells with value 1 are retained as 1, and others are set to 0.

    Example:
    >>> data = {'Label1': [1, 0, 1], 'Label2': [0, 1, 1]}
    >>> df_labels = pd.DataFrame(data)
    >>> calculate_true_labels_equal_to_one(df_labels)
       Label1  Label2
    0       1       0
    1       0       1
    2       1       1
    """
    return (true_labels == 1).astype(int)



########################################################################
#                                                                      #
########################################################################
def mlem_true_0(true_labels):
    """
    Returns a DataFrame indicating where the true labels are equal to 0.

    Parameters:
    -------
    true_labels (pd.DataFrame): The DataFrame containing the true labels with binary values (0 or 1).

    Returns:
    -------
    pd.DataFrame: A DataFrame of the same shape as `true_labels`, where cells with value 0 are retained as 1, and others are set to 0.

    Example:
    >>> data = {'Label1': [1, 0, 1], 'Label2': [0, 1, 1]}
    >>> df_labels = pd.DataFrame(data)
    >>> true_0(df_labels)
       Label1  Label2
    0       0       1
    1       1       0
    2       0       0
    """
    return (true_labels == 0).astype(int)



########################################################################
#                                                                      #
########################################################################
def mlem_pred_1(pred_labels):
    """
    Returns a DataFrame indicating where the predicted labels are equal to 1.

    Parameters:
    -------
    pred_labels (pd.DataFrame): The DataFrame containing the predicted labels with binary values (0 or 1).

    Returns:
    -------
    pd.DataFrame: A DataFrame of the same shape as `pred_labels`, where cells with value 1 are retained as 1, and others are set to 0.

    Example:
    >>> data = {'Label1': [1, 0, 1], 'Label2': [0, 1, 1]}
    >>> df_pred = pd.DataFrame(data)
    >>> pred_1(df_pred)
       Label1  Label2
    0       1       0
    1       0       1
    2       1       1
    """
    return (pred_labels == 1).astype(int)



########################################################################
#                                                                      #
########################################################################
def mlem_pred_0(pred_labels):
    """
    Returns a DataFrame indicating where the predicted labels are equal to 0.

    Parameters:
    -------
    pred_labels (pd.DataFrame): The DataFrame containing the predicted labels with binary values (0 or 1).

    Returns:
    -------
    pd.DataFrame: A DataFrame of the same shape as `pred_labels`, where cells with value 0 are retained as 1, and others are set to 0.

    Example:
    >>> data = {'Label1': [1, 0, 1], 'Label2': [0, 1, 1]}
    >>> df_pred = pd.DataFrame(data)
    >>> pred_0(df_pred)
       Label1  Label2
    0       0       1
    1       1       0
    2       0       0
    """
    return (pred_labels == 0).astype(int)



########################################################################
#                                                                      #
########################################################################
def mlem_total_true_1(true_1):
    """
    Returns a Series containing the total count of true labels equal to 1 for each label.

    Parameters:
    -------
    true_1 (pd.DataFrame): A DataFrame where cells with value 1 indicate true labels equal to 1, and 0s elsewhere.

    Returns:
    -------
    pd.Series: A Series containing the total count of true labels equal to 1 for each label (i.e., the sum of each column).

    Example:
    >>> data = {'Label1': [1, 0, 1], 'Label2': [0, 1, 1]}
    >>> df_true_1 = pd.DataFrame(data)
    >>> total_true_1(df_true_1)
    Label1    2
    Label2    2
    dtype: int64
    """
    return true_1.sum()



########################################################################
#                                                                      #
########################################################################
def mlem_total_true_0(true_0):
    """
    Returns a Series containing the total count of true labels equal to 0 for each label.

    Parameters:
    -------
    true_0 (pd.DataFrame): A DataFrame where cells with value 1 indicate true labels equal to 0, and 0s elsewhere.

    Returns:
    -------
    pd.Series: A Series containing the total count of true labels equal to 0 for each label (i.e., the sum of each column).

    Example:
    >>> data = {'Label1': [0, 1, 0], 'Label2': [1, 0, 0]}
    >>> df_true_0 = pd.DataFrame(data)
    >>> total_true_0(df_true_0)
    Label1    2
    Label2    2
    dtype: int64
    """
    return true_0.sum()


########################################################################
#                                                                      #
########################################################################
def mlem_total_pred_1(pred_1):
    """
    Returns a Series containing the total count of predicted labels equal to 1 for each label.

    Parameters:
    -------
    pred_1 (pd.DataFrame): A DataFrame where cells with value 1 indicate predicted labels equal to 1, and 0s elsewhere.

    Returns:
    -------
    pd.Series: A Series containing the total count of predicted labels equal to 1 for each label (i.e., the sum of each column).

    Example:
    >>> data = {'Label1': [1, 0, 1], 'Label2': [0, 1, 1]}
    >>> df_pred_1 = pd.DataFrame(data)
    >>> total_pred_1(df_pred_1)
    Label1    2
    Label2    2
    dtype: int64
    """
    return pred_1.sum()



########################################################################
#                                                                      #
########################################################################
def mlem_total_pred_0(pred_0):
    """
    Returns a Series containing the total count of predicted labels equal to 0 for each label.

    Parameters:
    -------
    pred_0 (pd.DataFrame): A DataFrame where cells with value 1 indicate predicted labels equal to 0, and 0s elsewhere.

    Returns:
    -------
    pd.Series: A Series containing the total count of predicted labels equal to 0 for each label (i.e., the sum of each column).

    Example:
    >>> data = {'Label1': [0, 1, 0], 'Label2': [1, 0, 1]}
    >>> df_pred_0 = pd.DataFrame(data)
    >>> total_pred_0(df_pred_0)
    Label1    2
    Label2    2
    dtype: int64
    """
    return pred_0.sum()





########################################################################
#                                                                      #
########################################################################
def mlem_tpi(true_1, pred_1):
    """
    Calculate True Positives (TP): The model predicted 1 and the correct response is 1.

    Parameters:
    -------
    true_1 (pd.DataFrame): A DataFrame where cells with value 1 indicate true labels equal to 1.
    
    pred_1 (pd.DataFrame): A DataFrame where cells with value 1 indicate predicted labels equal to 1.

    Returns:
    -------
    pd.DataFrame: A DataFrame indicating True Positives for each label.
    
    Example:
    >>> true_1 = pd.DataFrame({'Label1': [1, 0, 1], 'Label2': [1, 1, 0]})
    >>> pred_1 = pd.DataFrame({'Label1': [1, 0, 0], 'Label2': [1, 1, 0]})
    >>> calculate_tp(true_1, pred_1)
    Label1    1
    Label2    1
    dtype: int64
    """
    TPI = (true_1 & pred_1).astype(int)
    return TPI



########################################################################
#                                                                      #
########################################################################
def mlem_tni(true_0, pred_0):
    """
    Calculate True Negatives (TN): The model predicted 0 and the correct response is 0.

    Parameters:
    -------
    true_0 (pd.DataFrame): A DataFrame where cells with value 1 indicate true labels equal to 0.
    
    pred_0 (pd.DataFrame): A DataFrame where cells with value 1 indicate predicted labels equal to 0.

    Returns:
    -------
    pd.DataFrame: A DataFrame indicating True Negatives for each label.

    Example:
    >>> true_0 = pd.DataFrame({'Label1': [1, 0, 0], 'Label2': [0, 1, 1]})
    >>> pred_0 = pd.DataFrame({'Label1': [1, 0, 1], 'Label2': [0, 0, 1]})
    >>> calculate_tn(true_0, pred_0)
    Label1    2
    Label2    1
    dtype: int64
    """
    TNI = (true_0 & pred_0).astype(int)
    return TNI



########################################################################
#                                                                      #
########################################################################
def mlem_fpi(true_0, pred_1):
    """
    Calculate False Positives (FP): The model predicted 1 and the correct response is 0.

    Parameters:
    -------
    true_0 (pd.DataFrame): A DataFrame where cells with value 1 indicate true labels equal to 0.
    
    pred_1 (pd.DataFrame): A DataFrame where cells with value 1 indicate predicted labels equal to 1.

    Returns:
    -------
    pd.DataFrame: A DataFrame indicating False Positives for each label.

    Example:
    >>> true_0 = pd.DataFrame({'Label1': [1, 0, 0], 'Label2': [0, 1, 1]})
    >>> pred_1 = pd.DataFrame({'Label1': [1, 1, 0], 'Label2': [0, 1, 0]})
    >>> calculate_fp(true_0, pred_1)
    Label1    1
    Label2    0
    dtype: int64
    """
    FPI = (true_0 & pred_1).astype(int)
    return FPI



########################################################################
#                                                                      #
########################################################################
def mlem_fni(true_1, pred_0):
    """
    Calculate False Negatives (FN): The model predicted 0 and the correct response is 1.

    Parameters:
    -------
    true_1 (pd.DataFrame): A DataFrame where cells with value 1 indicate true labels equal to 1.
    
    pred_0 (pd.DataFrame): A DataFrame where cells with value 1 indicate predicted labels equal to 0.

    Returns:
    -------
    pd.DataFrame: A DataFrame indicating False Negatives for each label.

    Example:
    >>> true_1 = pd.DataFrame({'Label1': [1, 0, 1], 'Label2': [1, 1, 0]})
    >>> pred_0 = pd.DataFrame({'Label1': [0, 1, 0], 'Label2': [1, 0, 1]})
    >>> calculate_fn(true_1, pred_0)
    Label1    1
    Label2    1
    dtype: int64
    """
    FNI = (true_1 & pred_0).astype(int)
    return FNI



########################################################################
#                                                                      #
########################################################################
def mlem_tpl(TPi):
    """
    Calculate the total number of True Positives (TP) for each label.

    Parameters:
    -------
    TPi (pd.DataFrame): A DataFrame where cells with value 1 indicate True Positives.

    Returns:
    -------
    pd.Series: A Series with the total number of True Positives for each label.

    Example:
    >>> TPi = pd.DataFrame({'Label1': [1, 0, 1], 'Label2': [1, 1, 0]})
    >>> calculate_tp_totals(TPi)
    Label1    2
    Label2    2
    dtype: int64
    """
    TPL = TPi.sum()
    return TPL



########################################################################
#                                                                      #
########################################################################
def mlem_fpl(FPi):
    """
    Calculate the total number of False Positives (FP) for each label.

    Parameters:
    -------
    FPi (pd.DataFrame): A DataFrame where cells with value 1 indicate False Positives.

    Returns:
    -------
    pd.Series: A Series with the total number of False Positives for each label.

    Example:
    >>> FPi = pd.DataFrame({'Label1': [1, 0, 1], 'Label2': [0, 1, 1]})
    >>> calculate_fp_totals(FPi)
    Label1    2
    Label2    2
    dtype: int64
    """
    FPL = FPi.sum()
    return FPL



########################################################################
#                                                                      #
########################################################################
def mlem_fnl(FNi):
    """
    Calculate the total number of False Negatives (FN) for each label.

    Parameters:
    -------
    FNi (pd.DataFrame): A DataFrame where cells with value 1 indicate False Negatives.

    Returns:
    -------
    pd.Series: A Series with the total number of False Negatives for each label.

    Example:
    >>> FNi = pd.DataFrame({'Label1': [1, 0, 1], 'Label2': [1, 1, 0]})
    >>> calculate_fn_totals(FNi)
    Label1    2
    Label2    2
    dtype: int64
    """
    FNL = FNi.sum()
    return FNL




########################################################################
#                                                                      #
########################################################################
def mlem_tnl(TNi):
    """
    Calculate the total number of True Negatives (TN) for each label.

    Parameters:
    -------
    TNi (pd.DataFrame): A DataFrame where cells with value 1 indicate True Negatives.

    Returns:
    -------
    pd.Series: A Series with the total number of True Negatives for each label.

    Example:
    >>> TNi = pd.DataFrame({'Label1': [1, 0, 1], 'Label2': [0, 1, 1]})
    >>> calculate_tn_totals(TNi)
    Label1    2
    Label2    2
    dtype: int64
    """
    TNL = TNi.sum()
    return TNL



    
########################################################################
#                                                                      #
########################################################################
def mlem_confusion_matrix(true_labels, pred_labels):
    """
    Compute the multi-label confusion matrix and various performance metrics.

    This function calculates the confusion matrix components and derived metrics for multi-label classification.
    It generates a DataFrame containing the counts of True Positives (TP), True Negatives (TN), 
    False Positives (FP), and False Negatives (FN) for each label. Additionally, it computes the 
    percentage of correctly and incorrectly classified labels, and provides totals for columns and rows.

    Parameters:
    ----------
    true_labels : pd.DataFrame
        A DataFrame containing the true binary labels.
    
    pred_labels : pd.DataFrame
        A DataFrame containing the predicted binary labels.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with confusion matrix components, percentages, and totals.
    """

    # true labels = binary
    # pred labels = binary

    # Get the number of labels
    num_labels = number_of_labels(true_labels)    

    # Get the number of instances
    num_instances = number_of_instances(true_labels)
    
    # Get the count of positive instances for each label
    pi = positive_instances(true_labels)    

    # Get the count of negative instances for each label
    ni = negative_instances(true_labels, pi)    

    # Calculate where true labels are equal to 1
    true_1 = mlem_true_1(true_labels)

    # Calculate where true labels are equal to 0
    true_0 = mlem_true_0(true_labels)

    # Calculate where predicted labels are equal to 1
    pred_1 = mlem_pred_1(pred_labels)

    # Calculate where predicted labels are equal to 0
    pred_0 = mlem_pred_0(pred_labels)

     # Calculate the total count of true labels equal to 1 for each label
    total_true_1 = mlem_total_true_1(true_1)
    
    # Calculate the total count of true labels equal to 0 for each label
    total_true_0 = mlem_total_true_0(true_0)    

    # Calculate the total count of predicted labels equal to 1 for each label
    total_pred_1 = mlem_total_pred_1(pred_1)    

    # Calculate the total count of predicted labels equal to 0 for each label
    total_pred_0 = mlem_total_pred_0(pred_0)    

    # Calculate and print True Positives
    tpi = mlem_tpi(total_true_1, total_pred_1)    

    # Calculate and print True Negatives
    tni = mlem_tni(total_true_0, total_pred_0)    

    # Calculate and print False Positives
    fpi = mlem_fpi(total_true_0, total_pred_1)

    # Calculate and print False Negatives
    fni = mlem_fni(total_true_1, total_pred_0)
    
    # Calculate totals
    TPL = mlem_tpl(tpi)
    TNL = mlem_tnl(tni)
    FNL = mlem_fnl(fni)
    FPL = mlem_fpl(fpi)

    # Criar o DataFrame
    matrix = {
        'Total_True_Labels_1': total_true_1,
        'Total_True_Labels_0': total_true_0,
        'Total_Predicted_Labels_1': total_pred_1,
        'Total_Predicted_Labels_0': total_pred_0,
        'TPi': tpi,
        'TNi': tni,
        'FPi': fpi,
        'FNi': fni,
        'TP': TPL,
        'TN': TNL,
        'FN': FNL,
        'FP': FPL
    }

    matrix = pd.DataFrame(matrix)

    confusion_matrix_percentage = matrix / num_instances

    # Calculate total of incorrectly classified labels
    wrong_classifications = matrix['FP'] + matrix['FN']

    # Calculate percentage of incorrectly classified labels
    percent_wrong = wrong_classifications / num_instances

    # Calculate total of correctly classified labels
    correct_classifications = matrix['TP'] + matrix['TN']

    # Calculate percentage of correctly classified labels
    percent_correct = correct_classifications / num_instances

    # Combine all metrics into a result DataFrame    
    matrix['Correct'] = correct_classifications
    matrix['Percent_Correct'] = percent_correct
    matrix['Wrong'] = wrong_classifications
    matrix['Percent_Wrong'] = percent_wrong

    # Calculate totals for confusion matrix
    total_by_column = matrix.sum(axis=0)
    total_by_row = matrix.sum(axis=1)

    # Add total by column as a new row
    matrix.loc['Total_Columns'] = total_by_column

    # Add total by row as a new column
    matrix['Total_Rows'] = total_by_row

    # Supondo que co_ma seja o seu DataFrame
    matrix['labels'] = matrix.index  # Adiciona os nomes do índice como uma nova coluna 'labels'
    # Reorganizar as colunas para que 'labels' seja a primeira
    cols = ['labels'] + [col for col in matrix.columns if col != 'labels']  # Nova ordem de colunas
    matrix = matrix[cols]  # Reorganizar o DataFrame com 'labels' como a primeira coluna

    # Resetar o índice numérico
    matrix = matrix.reset_index(drop=True)

    return matrix



