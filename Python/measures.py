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

import numpy as np
import pandas as pd


#======================================================================#
# BIPARTITIONS                                                         #
#       INSTANCE BASED                                                 #
#               1. Accuracy                                            #
#               2. Subset Accuracy                                     #
#               3. Hamming Loss                                        #
#               4. 0/1 Loss                                            #
#               5. Precision                                           #
#               6. Recall                                              #
#               7. F1                                                  #
#       LABEL BASED                                                    #
#               1. Macro Precision                                     #
#               2. Macro Recall                                        #
#               3. Macro F1                                            #
#               4. Micro Precision                                     #
#               5. Micro Recall                                        #
#               6. Micro F1                                            #
#======================================================================#



#======================================================================#
#                                                                      #
# 'micro':                                                             #
#       Calculate metrics globally by counting the total true          #
#       positives, false negatives and false positives.                #
#                                                                      #
# 'macro':                                                             # 
#       Calculate metrics for each label, and find their unweighted    #
#       mean. This does not take label imbalance into account.         #
#                                                                      #
# 'weighted':                                                          # 
#       Calculate metrics for each label, and find their average       #
#       weighted by support (the number of true instances for each     #
#       label). This alters ‘macro’ to account for label imbalance;    #
#       it can result in an F-score that is not between precision      #
#       and recall.                                                    #
#                                                                      #
# 'samples':                                                           #
#       Calculate metrics for each instance, and find their average    #
#       (only meaningful for multilabel classification where this      #
#       differs from accuracy_score).                                  #
#                                                                      #
#======================================================================#



########################################################################
#                                                                      #
########################################################################
def mlem_accuracy(true_labels: pd.DataFrame, pred_labels: pd.DataFrame) -> float:
    """
    Calculate the accuracy for multi-label classification.

    Accuracy is defined as the proportion of correctly predicted labels out of the total number of labels.

    Parameters:
    -------
    true_labels (pd.DataFrame): The DataFrame containing the true labels with binary values (0 or 1).
    
    pred_labels (pd.DataFrame): The DataFrame containing the predicted labels with binary values (0 or 1).

    Returns:
    -------
    float: Accuracy value.

    References:
    -------
    [1] Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel Learning. 
    ACM Comput. Surv., 47(3), 52:1-52:38.   
    
    """
    # Ensure the DataFrames are in the same format
    if true_labels.shape != pred_labels.shape:
        raise ValueError("The shape of true_labels and pred_labels must be the same")

    # Compute the OR operation (true if either true_labels or pred_labels is 1)
    true_yi_or_pred_yi = np.logical_or(true_labels, pred_labels).astype(int)
    
    # Compute the AND operation (true if both true_labels and pred_labels are 1)
    true_and_pred_yi = np.logical_and(true_labels, pred_labels).astype(int)
    
    # Calculate total labels where either true or predicted label is 1 (denominator)
    total_1 = true_yi_or_pred_yi.sum(axis=1)
    
    # Calculate true positives (numerator)
    total_2 = true_and_pred_yi.sum(axis=1)
    
    # Handle division by zero: Set denominator to 1 where it is 0 to avoid invalid division
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = np.mean(np.divide(total_2, total_1, out=np.zeros_like(total_2, dtype=float), where=total_1 != 0))
    
    # Optional: Handle the case where all instances had no labels (total_1 is zero for all rows)
    if np.isnan(accuracy):
        accuracy = 0.0

    return accuracy


########################################################################
#                                                                      #
########################################################################
def mlem_subset_accuracy(true_labels: pd.DataFrame, pred_labels: pd.DataFrame) -> float:
    """
    Calculate the Subset Accuracy (Exact Match Ratio) for multi-label classification.

    Subset Accuracy measures the fraction of samples where the predicted labels exactly match the true labels.

    Parameters:
    -------
    true_labels (pd.DataFrame): The DataFrame containing the true labels with binary values (0 or 1).
    
    pred_labels (pd.DataFrame): The DataFrame containing the predicted labels with binary values (0 or 1).

    Returns:
    -------
    float: Subset Accuracy value.

    References:
    -------
    [1] Zhu, S., Ji, X., Xu, W., & Gong, Y. (2005). Multilabelled Classification 
    Using Maximum Entropy Method. In Proceedings of the 28th. Annual International 
    ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'05)
    (pp. 274-281).

    """
    # Ensure the DataFrames are in the same format
    if true_labels.shape != pred_labels.shape:
        raise ValueError("The shape of true_labels and pred_labels must be the same")
    
    # Handle empty DataFrames
    if true_labels.empty or pred_labels.empty:
        print("Warning: One or both input DataFrames are empty.")
        return 0.0

    # Compute whether each instance is exactly correct
    correct_predictions = (true_labels == pred_labels).all(axis=1)
    
    # Convert boolean results to integer (1 for True, 0 for False)
    correct_predictions_int = correct_predictions.astype(int)
    
    # Calculate subset accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        subset_accuracy = np.mean(correct_predictions_int)
    
    # Handle potential NaN result (e.g., if there are no instances)
    if np.isnan(subset_accuracy):
        subset_accuracy = 0.0

    return subset_accuracy





#======================================================================#
# RANKING                                                              #
#       1. Average Precision                                           #
#       2. Coverage                                                    #
#       3. Is Error                                                    #
#       4. Ranking Error                                               #
#       5. Margin Loss                                                 #
#       6. Ranking Loss                                                #
#       7. Margin Loss                                                 #
#======================================================================#



########################################################################
#                                                                      #
########################################################################
def mlem_ranking(pred_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the ranking of scores based on their values in descending order.
    The ranking is computed row-wise and is based on the inverse of the scores.

    Parameters:
    -------
    pred_scores (pd.DataFrame): A DataFrame containing the predicted probabilities for each label.

    Returns:
    -------
    pd.DataFrame: A DataFrame with the same shape as `scores` containing the ranks of the scores.
    
    Example:
    -------
    >>> scores = pd.DataFrame({
    ...     'Label1': [0.9, 0.1, 0.8],
    ...     'Label2': [0.3, 0.7, 0.6],
    ...     'Label3': [0.8, 0.2, 0.1]
    ... })
    >>> calculate_ranking(scores)
       Label1  Label2  Label3
    0       1       3       2
    1       3       1       2
    2       1       2       3
    """
    # Convert scores to numpy array for processing
    scores_np = pred_scores.values
    
    # Calculate ranks based on 1 - scores
    ranks = np.apply_along_axis(lambda row: np.argsort(np.argsort(-row)) + 1, axis=1, arr=1 - scores_np)
    
    # Convert ranks back to DataFrame
    ranking_df = pd.DataFrame(ranks, index=pred_scores.index, columns=pred_scores.columns)
    
    return ranking_df



########################################################################
#                                                                      #
########################################################################
def mlem_one_error(true_labels, pred_scores):
    """
    Compute the One Error metric for multi-label classification.

    The One Error metric measures the proportion of instances where the highest-ranked label is not in the set of true labels.

    Parameters:
    ----------
    true_labels : pd.DataFrame
        A DataFrame where each row represents a sample and each column represents a label.
        The values are binary (0 or 1) indicating the presence or absence of the label.
    
    pred_scores : pd.DataFrame
        A DataFrame where each row represents a sample and each column represents a label.
        The values are prediction scores for each label.

    Returns:
    -------
    float
        The One Error metric value.

    Example:
    -------
    >>> true_labels = pd.DataFrame([[1, 0, 0], [0, 1, 1], [1, 1, 0]], columns=['A', 'B', 'C'])
    >>> pred_scores = pd.DataFrame([[0.2, 0.5, 0.3], [0.4, 0.2, 0.6], [0.7, 0.1, 0.2]], columns=['A', 'B', 'C'])
    >>> mlem_one_error(true_labels, pred_scores)
    0.6666666666666666

    References:
    ----------
    Schapire, R. E., & Singer, Y. (2000). BoosTexter: A boosting-based system for text categorization. 
    Machine Learning, 39(2), 135-168.
    """
    # Convert DataFrames to numpy arrays for computation
    true_labels = true_labels.to_numpy()
    pred_scores = pred_scores.to_numpy()
    
    # Obtain ranking from prediction scores
    ranking = mlem_ranking(pd.DataFrame(pred_scores, columns=pred_scores.columns))
    
    # Determine the highest-ranked label for each sample
    predicted_labels = np.argmin(ranking.to_numpy(), axis=1)
    
    # Compute the One Error metric
    errors = np.array([1 if true_labels[i, predicted_labels[i]] == 0 else 0 for i in range(true_labels.shape[0])])
    oe = np.mean(errors)
    
    return oe






########################################################################
#                                                                      #
########################################################################
def mlem_average_precision(true_labels: pd.DataFrame, pred_scores: pd.DataFrame) -> float:
    """
    Calculate the Average Precision for multi-label classification based on true labels and predicted scores.

    Parameters:
    -------
    true_labels (pd.DataFrame): A DataFrame with binary values indicating the true labels for each instance.
    
    pred_scores (pd.DataFrame): A DataFrame with predicted scores for each label for each instance.

    Returns:
    -------
    float: The Average Precision score.

    References:
    -------
    Tsoumakas, K., et al. (2009). Multi-Label Classification with Label Constraints. 
    In Proceedings of the ECML PKDD 2008 Workshop on Preference Learning (PL-08, Antwerp, Belgium), 157-171.

    Example:
    -------
    >>> true_labels = pd.DataFrame({
    ...     'L1': [1, 0, 1, 1],
    ...     'L2': [0, 1, 1, 0],
    ...     'L3': [0, 0, 1, 1],
    ...     'L4': [1, 1, 0, 1]
    ... })
    >>> pred_scores = pd.DataFrame({
    ...     'L1': [0.9, 0.1, 0.8, 0.5],
    ...     'L2': [0.4, 0.7, 0.6, 0.2],
    ...     'L3': [0.6, 0.2, 0.7, 0.3],
    ...     'L4': [0.5, 0.3, 0.8, 0.6]
    ... })
    >>> average_precision(true_labels, pred_scores)
    0.5833333333333334
    """
    ranking = mlem_ranking(pred_scores)
    Y = true_labels.values
    Yi = np.sum(Y, axis=1)
    
    non_empty_indices = np.where(Yi > 0)[0]
    Y_filtered = Y[non_empty_indices, :]
    ranking_filtered = ranking.values[non_empty_indices, :]
    
    def compute_average_precision_for_instance(instance_labels, instance_ranking):
        ap_sum = 0.0
        num_relevant_labels = np.sum(instance_labels)
        for label in np.where(instance_labels == 1)[0]:
            rank = instance_ranking[label]
            relevant_labels_at_rank = np.sum(instance_ranking <= rank)
            ap_sum += relevant_labels_at_rank / rank
        return ap_sum / num_relevant_labels if num_relevant_labels > 0 else 0.0
    
    ap_scores = np.array([compute_average_precision_for_instance(Y_filtered[i], ranking_filtered[i])
                          for i in range(len(Y_filtered))])
    
    return np.mean(ap_scores)




########################################################################
#                                                                      #
########################################################################
def mlem_precision_at_k(true_labels: pd.DataFrame, pred_scores: pd.DataFrame) -> float:
    """
    Calculate the Precision at k for multi-label classification based on true labels and predicted scores.

    Parameters:
    -------
    true_labels (pd.DataFrame): A DataFrame with binary values indicating the true labels for each instance.
    
    pred_scores (pd.DataFrame): A DataFrame with predicted scores for each label for each instance.

    Returns:
    -------
    float: The Precision at k score.

    References:
    -------
    Schapire, R. E., & Singer, Y. (2000). BoosTexter: A boosting-based system for text categorization. 
    Machine Learning, 39(2), 135-168.

    Example:
    -------
    >>> true_labels = pd.DataFrame({
    ...     'L1': [1, 0, 1, 1],
    ...     'L2': [0, 1, 1, 0],
    ...     'L3': [0, 0, 1, 1],
    ...     'L4': [1, 1, 0, 1]
    ... })
    >>> pred_scores = pd.DataFrame({
    ...     'L1': [0.9, 0.1, 0.8, 0.5],
    ...     'L2': [0.4, 0.7, 0.6, 0.2],
    ...     'L3': [0.6, 0.2, 0.7, 0.3],
    ...     'L4': [0.5, 0.3, 0.8, 0.6]
    ... })
    >>> precision_at_k(true_labels, pred_scores)
    0.6111111111111112
    """
    ranking = mlem_ranking(pred_scores)
    Y = true_labels.values
    Yi = np.sum(Y, axis=1)
    
    non_empty_indices = np.where(Yi > 0)[0]
    Y_filtered = Y[non_empty_indices, :]
    ranking_filtered = ranking.values[non_empty_indices, :]
    
    def compute_precision_at_k(instance_labels, instance_ranking):
        rks = instance_ranking[instance_labels == 1]
        precision_sum = sum(np.sum(instance_ranking <= r) / r for r in rks)
        return precision_sum / np.sum(instance_labels) if np.sum(instance_labels) > 0 else 0.0
    
    precision_at_k_scores = np.array([compute_precision_at_k(Y_filtered[i], ranking_filtered[i])
                                      for i in range(len(Y_filtered))])
    
    return np.mean(precision_at_k_scores)




########################################################################
#                                                                      #
########################################################################
def mlem_is_error(true_labels: pd.DataFrame, pred_scores: pd.DataFrame) -> float:
    """
    Calculate the Is Error metric to evaluate if the predicted ranking matches the true ranking.
    
    Parameters:
    -------
    true_labels (pd.DataFrame): A DataFrame containing the true labels for each instance.
    
    pred_scores (pd.DataFrame): A DataFrame containing the predicted scores for each label and instance.

    Returns:
    -------
    float: The Is Error metric value.
    
    Raises:
    -------
    ValueError: If the `true_labels` or `pred_scores` arguments are not provided.
    
    References:
    -------
    Crammer, K., & Singer, Y. (2003). A Family of Additive Online Algorithms for Category Ranking. 
    Journal of Machine Learning Research, 3(6), 1025-1058.
    """

    # Calculate rankings
    true_ranking = mlem_ranking(true_labels)
    predicted_ranking = mlem_ranking(pred_scores)

    # Ensure that true_ranking and predicted_ranking have the same shape
    if true_ranking.shape != predicted_ranking.shape:
        raise ValueError("The shapes of true_ranking and predicted_ranking must be the same.")
    
    # Calculate the Is Error metric
    rank_diff = np.abs(true_ranking.values - predicted_ranking.values)
    error_metric = np.mean(np.any(rank_diff != 0, axis=1))
    
    return error_metric



########################################################################
#                                                                      #
########################################################################
def compute_mloss_for_instance(true_labels_row, pred_ranking_row):
    """
    Compute the Margin Loss for a single instance.
    
    Parameters:
    -------
    true_labels_row (np.array): Binary array indicating the true labels for a single instance.
    
    pred_ranking_row (np.array): Ranking array indicating the predicted ranks for a single instance.
    
    Returns:
    -------
    float: The Margin Loss for the instance.
    """
    idxY = true_labels_row == 1
    if np.any(idxY):  # Check if there are any positive labels
        max_positive_rank = np.max(pred_ranking_row[idxY])
        min_negative_rank = np.min(pred_ranking_row[~idxY])
        return max(0, max_positive_rank - min_negative_rank)
    else:
        return 0
    


########################################################################
#                                                                      #
########################################################################
def mlem_margin_loss(true_labels: pd.DataFrame, pred_scores: pd.DataFrame) -> float:
    """
    Calculate the Margin Loss metric for multi-label classification.
    
    The Margin Loss metric quantifies the number of positions between positive and negative labels
    in the ranking. It measures the worst-case ranking difference between the highest-ranked positive
    label and the lowest-ranked negative label.

    Parameters:
    -------
    true_labels (pd.DataFrame): A DataFrame with binary values indicating the true labels for each instance.
    
    pred_scores (pd.DataFrame): A DataFrame with predicted scores for each label for each instance.

    Returns:
    -------
    float: The Margin Loss metric value.

    References:
    -------
    Loza Mencia, E., & Furnkranz, J. (2010). Efficient Multilabel Classification Algorithms for Large-Scale Problems in the Legal Domain.
    In Semantic Processing of Legal Texts (pp. 192-215).
   
    Example:
    -------
    >>> true_labels = pd.DataFrame({
    ...     'L1': [1, 0, 1, 1],
    ...     'L2': [0, 1, 1, 0],
    ...     'L3': [0, 0, 1, 1],
    ...     'L4': [1, 1, 0, 1]
    ... })
    >>> pred_scores = pd.DataFrame({
    ...     'L1': [0.9, 0.1, 0.8, 0.5],
    ...     'L2': [0.4, 0.7, 0.6, 0.2],
    ...     'L3': [0.6, 0.2, 0.7, 0.3],
    ...     'L4': [0.5, 0.3, 0.8, 0.6]
    ... })
    >>> margin_loss(true_labels, pred_scores)
    0.5
    """
    # Calculate rankings
    pred_ranking = mlem_ranking(pred_scores)
    
    # Ensure that true_labels and pred_ranking have the same shape
    if true_labels.shape != pred_ranking.shape:
        raise ValueError("The shapes of true_labels and pred_ranking must be the same.")
    
    mloss_values = [
        compute_mloss_for_instance(true_labels.iloc[i].values, pred_ranking.iloc[i].values)
        for i in range(len(true_labels))
    ]
    
    return np.mean(mloss_values)


########################################################################
#                                                                      #
########################################################################
def mlem_ranking_error(true_labels: pd.DataFrame, pred_scores: pd.DataFrame) -> float:
    """
    Calculate the Ranking Error (RE) metric for multi-label classification.

    The Ranking Error metric measures the sum of squared differences in positions 
    of predicted ranks versus true ranks. If the predicted ranking matches the true ranking exactly, RE = 0. 
    If the ranking is completely inverted, RE = 1.

    Parameters:
    -------
    true_labels (pd.DataFrame): A DataFrame with binary values indicating the true labels for each instance.
    
    pred_scores (pd.DataFrame): A DataFrame with predicted scores for each label for each instance.
    
    Returns:
    -------
    float: The Ranking Error metric value.

    References:
    -------
    Park, S.-H., & Furnkranz, J. (2008). Multi-Label Classification with Label Constraints. 
    Proceedings of the ECML PKDD 2008 Workshop on Preference Learning (PL-08, Antwerp, Belgium), 157-171.

    Example:
    -------
    >>> true_labels = pd.DataFrame({
    ...     'L1': [1, 0, 1, 1],
    ...     'L2': [0, 1, 1, 0],
    ...     'L3': [0, 0, 1, 1],
    ...     'L4': [1, 1, 0, 1]
    ... })
    >>> pred_scores = pd.DataFrame({
    ...     'L1': [0.9, 0.1, 0.8, 0.5],
    ...     'L2': [0.4, 0.7, 0.6, 0.2],
    ...     'L3': [0.6, 0.2, 0.7, 0.3],
    ...     'L4': [0.5, 0.3, 0.8, 0.6]
    ... })
    >>> ranking_error(true_labels, pred_scores)
    0.5
    """
    # Calculate rankings
    true_ranking = mlem_ranking(true_labels)
    pred_ranking = mlem_ranking(pred_scores)
    
    # Ensure that true_ranking and pred_ranking have the same shape
    if true_ranking.shape != pred_ranking.shape:
        raise ValueError("The shapes of true_ranking and pred_ranking must be the same.")
    
    # Calculate the Ranking Error metric
    differences = np.square(true_ranking.values - pred_ranking.values)
    error_metric = np.mean(np.sum(differences, axis=1))
    
    return error_metric



#======================================================================#
# LABEL PROBLEM                                                        #
#       1. CLP                                                         #
#       2. MLP                                                         #
#       3. WLP                                                         #
#======================================================================#


########################################################################
#                                                                      #
########################################################################
def mlem_clp(confusion_matrix: pd.DataFrame) -> float:
    """
    Calculate the Constant Label Problem (CLP) for multi-label classification.

    Parameters:
    ----------
    confusion_matrix : pd.DataFrame
        DataFrame containing the confusion matrix with columns 'TNL' (True Negatives) 
        and 'FNL' (False Negatives) for each label.

    Returns:
    -------
    float
        The CLP score, which represents the proportion of labels where TN + FN == 0.
    
    Reference:
    ----------
    [1] Rivolli, A., Soares, C., & Carvalho, A. C. P. de L. F. de. (2018). 
        Enhancing multilabel classification for food truck recommendation. 
        Expert Systems. Wiley-Blackwell. DOI: 10.1111/exsy.12304

    Example:
    -------
    >>> confusion_matrix = pd.DataFrame({
    ...     'TNL': [0, 2, 3, 0],
    ...     'FNL': [0, 1, 0, 0]
    ... })
    >>> calculate_clp(confusion_matrix)
    0.5
    """
    # Calculate (TN + FN) for each label
    tn_fn_sum = confusion_matrix['TN'] + confusion_matrix['FN']
    
    # Identify labels where (TN + FN) == 0
    clp_labels = (tn_fn_sum == 0).sum()
    
    # Number of labels
    total_labels = len(confusion_matrix)
    
    # Calculate CLP
    clp_score = clp_labels / total_labels
    
    return clp_score



########################################################################
#                                                                      #
########################################################################
def mlem_mlp(confusion_matrix: pd.DataFrame) -> float:
    """
    Calculate the Missing Label Prediction (MLP) for multi-label classification.

    Parameters:
    ----------
    confusion_matrix : pd.DataFrame
        DataFrame containing the confusion matrix with columns 'TPI', 'TNI', 'FPI', 'FNI',
        'TPL', 'TNL', 'FPL', 'FNL'.

    Returns:
    -------
    float
        The MLP score, representing the proportion of labels that are never predicted.
    
    Reference:
    ----------
    [1] Rivolli, A., Soares, C., & Carvalho, A. C. P. de L. F. de. (2018). 
        Enhancing multilabel classification for food truck recommendation. 
        Expert Systems. Wiley-Blackwell. DOI: 10.1111/exsy.12304
    
    Example:

    Example:
    -------
    >>> confusion_matrix = pd.DataFrame({
    ...     'TPI': [2, 2, 0, 2],
    ...     'TNI': [0, 2, 2, 0],
    ...     'FPI': [0, 2, 0, 0],
    ...     'FNI': [2, 2, 2, 2],
    ...     'TPL': [6, 6, 6, 6],
    ...     'TNL': [4, 4, 4, 4],
    ...     'FPL': [2, 2, 2, 2],
    ...     'FNL': [8, 8, 8, 8]
    ... })
    >>> calculate_mlp(confusion_matrix)
    0.5  
    """
    # Compute the total number of labels
    l = confusion_matrix.shape[0]
    
    # Calculate if TPI + FPI is zero for each label
    res = ((confusion_matrix['TPi'] + confusion_matrix['FPi']) == 0).astype(int)
    
    # Sum the results to get the count of labels that are never predicted
    res_2 = res.sum()
    
    # Calculate MLP as the proportion of such labels
    mlp = res_2 / l

    return mlp



########################################################################
#                                                                      #
########################################################################
def mlem_wlp(confusion_matrix: pd.DataFrame) -> float:
    """
    Calculate the Wrong Label Prediction (WLP) for multi-label classification.

    WLP measures when a label may be predicted for some instances, but these predictions are always wrong.

    Parameters:
    ----------
    confusion_matrix : pd.DataFrame
        DataFrame containing the confusion matrix with columns 'TPI', 'TNI', 'FPI', 'FNI',
        'TPL', 'TNL', 'FPL', 'FNL'.

    Returns:
    -------
    float
        The WLP score, representing the proportion of labels where the predictions are always wrong.

    Reference:
    ----------
    [1] Rivolli, A., Soares, C., & Carvalho, A. C. P. de L. F. de. (2018). 
        Enhancing multilabel classification for food truck recommendation. 
        Expert Systems. Wiley-Blackwell. DOI: 10.1111/exsy.12304
    
    Example:
    -------
    >>> confusion_matrix = pd.DataFrame({
    ...     'TPI': [2, 2, 0, 2],
    ...     'TNI': [0, 2, 2, 0],
    ...     'FPI': [0, 2, 0, 0],
    ...     'FNI': [2, 2, 2, 2],
    ...     'TPL': [6, 6, 6, 6],
    ...     'TNL': [4, 4, 4, 4],
    ...     'FPL': [2, 2, 2, 2],
    ...     'FNL': [8, 8, 8, 8]
    ... })
    >>> calculate_wlp(confusion_matrix)
    0.5
    """
    # Compute the total number of labels
    l = confusion_matrix.shape[0]
    
    # Calculate if TPL is zero for each label
    res_2 = (confusion_matrix['TP'] == 0).astype(int)
    
    # Sum the results to get the count of labels where predictions are always wrong
    wlp = res_2.sum() / l

    return wlp


