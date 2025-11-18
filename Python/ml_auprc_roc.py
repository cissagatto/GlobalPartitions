##############################################################################
# Copyright (C) 2024                                                         #
#                                                                            #
# CC BY-NC-SA 4.0                                                            #
#                                                                            #
# Canonical URL https://creativecommons.org/licenses/by-nc-sa/4.0/           #
# Attribution-NonCommercial-ShareAlike 4.0 International CC BY-NC-SA 4.0     #
#                                                                            #
# Prof. Elaine Cecilia Gatto | Prof. Ricardo Cerri | Prof. Mauri Ferrandin   #
#                                                                            #
# Federal University of São Carlos - UFSCar - https://www2.ufscar.br         #
# Campus São Carlos - Computer Department - DC - https://site.dc.ufscar.br   #
# Post Graduate Program in Computer Science - PPGCC                          # 
# http://ppgcc.dc.ufscar.br - Bioinformatics and Machine Learning Group      #
# BIOMAL - http://www.biomal.ufscar.br                                       #
#                                                                            #
# You are free to:                                                           #
#     Share — copy and redistribute the material in any medium or format     #
#     Adapt — remix, transform, and build upon the material                  #
#     The licensor cannot revoke these freedoms as long as you follow the    #
#       license terms.                                                       #
#                                                                            #
# Under the following terms:                                                 #
#   Attribution — You must give appropriate credit , provide a link to the   #
#     license, and indicate if changes were made . You may do so in any      #
#     reasonable manner, but not in any way that suggests the licensor       #
#     endorses you or your use.                                              #
#   NonCommercial — You may not use the material for commercial purposes     #
#   ShareAlike — If you remix, transform, or build upon the material, you    #
#     must distribute your contributions under the same license as the       #
#     original.                                                              #
#   No additional restrictions — You may not apply legal terms or            #
#     technological measures that legally restrict others from doing         #
#     anything the license permits.                                          #
#                                                                            #
##############################################################################

import os
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

# ------------------------------------------------------------------------------#
#                          ROBUST MACRO ROC AUC                                 #
# ------------------------------------------------------------------------------#
def robust_macro_roc_auc(true, pred, verbose=True):
    """
    Compute a robust macro-average ROC AUC across multiple binary labels,
    with interpolation and explicit handling of degenerate or undefined cases.

    ----------------------------------------------------------------------
    Description
    ----------------------------------------------------------------------
    For each label (column) in `true` and `pred` this function:
      - attempts to compute the ROC curve (FPR, TPR) and AUC using
        sklearn.metrics when possible;
      - handles special cases where `y_true` has no class variation
        (all zeros or all ones) or where predictions are constant or binary;
      - assigns deterministic AUC values for such edge cases to ensure
        numerical stability and interpretability;
      - returns both the simple macro-average of per-label AUCs and the AUC
        of the interpolated mean ROC curve.

    ----------------------------------------------------------------------
    Objective
    ----------------------------------------------------------------------
    Provide a stable and interpretable macro-averaged ROC-AUC metric for
    multi-label problems, even when some labels cannot produce a valid ROC
    curve under the standard definition (e.g., labels with no positive
    examples). The interpolated mean ROC offers an alternative aggregate
    summary by averaging TPRs on a common FPR grid.

    ----------------------------------------------------------------------
    Parameters
    ----------------------------------------------------------------------
    true : pandas.DataFrame
        Ground-truth binary indicators (0 or 1) for each label (columns).
        Each column represents a distinct label and all columns must have
        the same number of rows.

    pred : pandas.DataFrame
        Predicted scores or probabilities for each label (columns must match
        `true` and be in the same order). May contain binary values (0/1)
        or continuous probabilities in [0, 1].

    verbose : bool, default=True
        If True, prints a per-label message describing the handled case and
        the resulting AUC, and prints final macro statistics.

    ----------------------------------------------------------------------
    Returns
    ----------------------------------------------------------------------
    tuple: (fpr_dict_macro, tpr_dict_macro, macro_auc, macro_auc_interp, macro_auc_df)

    fpr_dict_macro : dict
        Mapping from label name to the array of False Positive Rates used
        to build the ROC for that label. For special-case labels, returns
        standardized arrays (e.g., [0.0, 1.0]).

    tpr_dict_macro : dict
        Mapping from label name to the array of True Positive Rates for that
        label. For special-case labels, returns standardized arrays such as
        [0.0, 1.0] or [0.0, 0.0] depending on the case.

    macro_auc : float
        The simple arithmetic mean of the per-label AUC values (including
        substituted AUCs for degenerate labels).

    macro_auc_interp : float
        The AUC computed from the interpolated mean ROC curve. The mean ROC
        curve is obtained by:
          1) forming a common set of FPR points across labels,
          2) interpolating each label's TPR onto that grid,
          3) averaging the interpolated TPRs and computing AUC over the grid.

    macro_auc_df : pandas.DataFrame
        DataFrame with one row per label and columns:
          - "Label": label name
          - "AUC": computed or assigned AUC value
          - "FPR": array of False Positive Rates (object dtype)
          - "TPR": array of True Positive Rates (object dtype)

    ----------------------------------------------------------------------
    Special-case assignment summary
    ----------------------------------------------------------------------
      - y_true all 0 and y_pred all 0  → AUC = 1.0
      - y_true all 1 and y_pred all 1  → AUC = 1.0
      - y_true all 0 and y_pred all 1  → AUC = 0.0
      - y_true all 1 and y_pred all 0  → AUC = 0.0
      - y_true constant (0 or 1) and y_pred probabilistic (not only 0/1)
        → AUC = 0.5
      - y_true all 1 and y_pred mixed {0,1} → AUC = 1.0
      - y_true all 0 and y_pred mixed {0,1} → AUC = 0.0
      - sklearn ROC computation error (ValueError) → AUC = 0.0
      - other fallback cases → AUC = 0.5

    ----------------------------------------------------------------------
    Example
    ----------------------------------------------------------------------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.datasets import make_multilabel_classification
    >>> # Generate example multi-label data
    >>> X, Y = make_multilabel_classification(n_samples=100, n_features=10,
    ...                                       n_classes=3, random_state=0)
    >>> true = pd.DataFrame(Y, columns=["Label1", "Label2", "Label3"])
    >>> rng = np.random.default_rng(0)
    >>> pred = pd.DataFrame({
    ...     "Label1": rng.random(100),
    ...     "Label2": rng.random(100),
    ...     "Label3": rng.random(100)
    ... })
    >>> fpr_dict, tpr_dict, macro_auc, macro_auc_interp, macro_df = robust_macro_roc_auc(true, pred, verbose=False)
    >>> print(f"Macro AUC (mean): {macro_auc:.4f}")
    >>> print(f"Macro AUC (interpolated): {macro_auc_interp:.4f}")
    >>> print(macro_df[["Label", "AUC"]])
    """

    fpr_dict_macro = {}
    tpr_dict_macro = {}
    macro_auc_df = []

    if verbose:
        print("\n" + "="*40)
        print("MACRO-AVERAGE ROC")
        print("="*40)

    # --- Iterate over each label (column) ---
    for col in true.columns:
        y_true = true[col].values
        y_pred = pred[col].values

        # --- CASE 1: y_true has only one class (no variation) ---
        if len(np.unique(y_true)) < 2:

            # (i) both empty: True=0, Pred=0, AUC = 1
            if np.all(y_true == 0) and np.all(y_pred == 0):
                auc = 1.0
                fpr_dict_macro[col] = np.array([0.0, 1.0])
                tpr_dict_macro[col] = np.array([0.0, 1.0])
                msg = f"{col}: True == 0 and PRED == 0 --> AUC = 1"

            # (ii) both full: True=1, Pred=1, AUC = 1
            elif np.all(y_true == 1) and np.all(y_pred == 1):
                auc = 1.0
                fpr_dict_macro[col] = np.array([0.0, 1.0])
                tpr_dict_macro[col] = np.array([0.0, 1.0])
                msg = f"{col}: True == 1 and PRED == 1 --> AUC = 1"

            # (iii) True empty, Pred full, AUC = 0
            elif np.all(y_true == 0) and np.all(y_pred == 1):
                auc = 0.0
                fpr_dict_macro[col] = np.array([0.0, 1.0])
                tpr_dict_macro[col] = np.array([1.0, 1.0])
                msg = f"{col}: True == 0 and PRED == 1 --> AUC = 0"

            # (iv) True full, Pred empty, AUC = 0
            elif np.all(y_true == 1) and np.all(y_pred == 0):
                auc = 0.0
                fpr_dict_macro[col] = np.array([0.0, 1.0])
                tpr_dict_macro[col] = np.array([0.0, 0.0])
                msg = f"{col}: True == 1 and PRED == 0 --> AUC = 0"

            # (v) True all 0 or all 1, PRED has probabilities, AUC = 9.5
            elif len(np.unique(y_true)) == 1 and not np.all(np.isin(y_pred, [0, 1])):
                auc = 0.5
                fpr_dict_macro[col] = np.array([0.0, 1.0])
                tpr_dict_macro[col] = np.array([0.0, 1.0])
                true_val = int(y_true[0])  # vai ser 0 ou 1
                msg = f"{col}: True == {true_val} and PRED probabilistic --> AUC = 0.5"               

            # (vi) True=1, Pred mixed (0/1), AUC = 1
            elif np.all(y_true == 1) and np.any(y_pred == 0) and np.any(y_pred == 1):
                auc = 1.0
                fpr_dict_macro[col] = np.array([0.0, 1.0])
                tpr_dict_macro[col] = np.array([0.0, 1.0])
                msg = f"{col}: True == 1 and PRED == 0/1 --> AUC = 1"

            # (vii) True=0, Pred mixed (0/1), AUC = 0
            elif np.all(y_true == 0) and np.any(y_pred == 0) and np.any(y_pred == 1):
                auc = 0.0
                fpr_dict_macro[col] = np.array([0.0, 1.0])
                tpr_dict_macro[col] = np.array([0.0, 0.0])
                msg = f"{col}: True == 0 and PRED == 0/1 --> AUC = 0"

            # fallback (shouldn't happen)
            else:
                auc = 0.5
                fpr_dict_macro[col] = np.array([0.0, 1.0])
                tpr_dict_macro[col] = np.array([0.0, 1.0])
                msg = f"{col}: Other case --> AUC = 0.5"

            if verbose:
                print(msg)

        # --- CASE 2: normal ROC ---
        else:
            try:
                fpr, tpr, _ = metrics.roc_curve(y_true.astype(float), y_pred.astype(float))
                auc = metrics.auc(fpr, tpr)
                fpr_dict_macro[col] = fpr
                tpr_dict_macro[col] = tpr
                if verbose:
                    print(f"{col}: Normal case --> AUC = {auc:.4f}")
            except ValueError:
                fpr_dict_macro[col] = np.array([0.0, 1.0])
                tpr_dict_macro[col] = np.array([0.0, 0.0])
                auc = 0.0
                if verbose:
                    print(f"{col}: ROC computation failed --> AUC = 0")

        macro_auc_df.append((col, auc))
        
    # --- Convert AUC results to DataFrame ---
    macro_auc_df = pd.DataFrame(
        [(label, 
          auc,
          fpr_dict_macro[label], 
          tpr_dict_macro[label])
         for label, auc in macro_auc_df],
        columns=["Label", "AUC", "FPR", "TPR"]
    )
    # --- Mean of AUCs ---
    macro_auc = np.mean(macro_auc_df["AUC"])

    # --- Interpolated mean ROC ---
    all_fpr = np.unique(np.concatenate([fpr_dict_macro[c] for c in fpr_dict_macro]))
    mean_tpr = np.zeros_like(all_fpr, dtype=float)

    for c in fpr_dict_macro:
        mean_tpr += np.interp(all_fpr, fpr_dict_macro[c], tpr_dict_macro[c])

    mean_tpr /= len(fpr_dict_macro)
    macro_auc_interp = metrics.auc(all_fpr, mean_tpr)

    if verbose:
        print("\n" + "-"*40)
        print(f"Macro AUC mean of individual AUCs: {macro_auc:.4f}")
        print(f"Macro AUC interpolated mean curve: {macro_auc_interp:.4f}")
        print("="*40)

    return fpr_dict_macro, tpr_dict_macro, macro_auc, macro_auc_interp, macro_auc_df



# ------------------------------------------------------------------------------#
#                          ROBUST MICRO ROC AUC                                 #
# ------------------------------------------------------------------------------#
def robust_micro_roc_auc(true, pred, verbose=True):
    """
    Compute a robust micro-average ROC-AUC score with special-case handling.

    ------------------------------------------------------------
    Description
    ------------------------------------------------------------
    This function calculates the micro-averaged ROC-AUC score for multilabel
    classification tasks, with robust handling of degenerate cases such as:
    - All labels being 0 or 1.
    - Model predicting a single constant value.
    - Lack of positive or negative samples.
    - Probabilistic predictions when the ground truth is constant.

    Unlike `sklearn.metrics.roc_auc_score`, this implementation avoids NaN
    or undefined results by explicitly assigning AUC values (0.0, 0.5, or 1.0)
    in these special scenarios.

    ------------------------------------------------------------
    Objective
    ------------------------------------------------------------
    Provide a consistent and interpretable AUC_micro value even when
    some edge cases occur — particularly useful in highly imbalanced
    multilabel problems or small datasets.

    ------------------------------------------------------------
    Parameters
    ------------------------------------------------------------
    true : pandas.DataFrame
        Ground-truth binary labels of shape (n_samples, n_classes).

    pred : pandas.DataFrame
        Predicted probabilities or binary predictions of the same shape.

    verbose : bool, optional (default=True)
        If True, prints diagnostic messages about which special case
        was applied and the computed AUC value.

    ------------------------------------------------------------
    Returns
    ------------------------------------------------------------
    fpr_micro : numpy.ndarray
        False Positive Rate values for the micro-averaged ROC curve.

    tpr_micro : numpy.ndarray
        True Positive Rate values for the micro-averaged ROC curve.

    auc_micro : float
        The computed micro-average AUC value. Guaranteed to be numeric
        and well-defined (never NaN).

    ------------------------------------------------------------
    Example
    ------------------------------------------------------------
    >>> import pandas as pd
    >>> true = pd.DataFrame({
    ...     'Label1': [0, 1, 1, 0],
    ...     'Label2': [1, 0, 1, 0]
    ... })
    >>> pred = pd.DataFrame({
    ...     'Label1': [0.1, 0.9, 0.8, 0.2],
    ...     'Label2': [0.7, 0.3, 0.9, 0.1]
    ... })
    >>> fpr_micro, tpr_micro, auc_micro = robust_micro_roc_auc(true, pred)
    >>> print(f"Micro-average AUC: {auc_micro:.4f}")
    ------------------------------------------------------------
    """

    # ----------------------------------------------------------------------
    # Flatten all labels into a single vector (treat every (sample, label) pair)
    # ----------------------------------------------------------------------
    y_true_all = true.values.ravel().astype(float)
    y_pred_all = pred.values.ravel().astype(float)

    if verbose:
        print("\n" + "=" * 40)
        print("MICRO-AVERAGE ROC")
        print("=" * 40)

    # --- CASE 1: y_true has only one class (no variation) ---
    if len(np.unique(y_true_all)) < 2:

        # (i) both empty: True=0, Pred=0, AUC = 1
        if np.all(y_true_all == 0) and np.all(y_pred_all == 0):
            auc_micro = 1.0
            fpr_micro = np.array([0.0, 1.0])
            tpr_micro = np.array([0.0, 1.0])
            msg = "True == 0 and PRED == 0 --> AUC = 1"

        # (ii) both full: True=1, Pred=1, AUC = 1
        elif np.all(y_true_all == 1) and np.all(y_pred_all == 1):
            auc_micro = 1.0
            fpr_micro = np.array([0.0, 1.0])
            tpr_micro = np.array([0.0, 1.0])
            msg = "True == 1 and PRED == 1 --> AUC = 1"

        # (iii) True empty, Pred full, AUC = 0
        elif np.all(y_true_all == 0) and np.all(y_pred_all == 1):
            auc_micro = 0.0
            fpr_micro = np.array([0.0, 1.0])
            tpr_micro = np.array([1.0, 1.0])
            msg = "True == 0 and PRED == 1 --> AUC = 0"

        # (iv) True full, Pred empty, AUC = 0
        elif np.all(y_true_all == 1) and np.all(y_pred_all == 0):
            auc_micro = 0.0
            fpr_micro = np.array([0.0, 1.0])
            tpr_micro = np.array([0.0, 0.0])
            msg = "True == 1 and PRED == 0 --> AUC = 0"

        # (v) True all 0 or all 1, PRED has probabilities, AUC = 9.5
        elif len(np.unique(y_true_all)) == 1 and not np.all(np.isin(y_pred_all, [0, 1])):
            auc_micro = 0.5
            fpr_micro = np.array([0.0, 1.0])
            tpr_micro = np.array([0.0, 1.0])
            true_val = int(y_true_all[0])
            msg = f"True == {true_val} and PRED probabilistic --> AUC = 0.5"

        # (vi) True=1, Pred mixed (0/1), AUC = 1
        elif np.all(y_true_all == 1) and np.any(y_pred_all == 0) and np.any(y_pred_all == 1):
            auc_micro = 1.0
            fpr_micro = np.array([0.0, 1.0])
            tpr_micro = np.array([0.0, 1.0])
            msg = "True == 1 and PRED == 0/1 --> AUC = 1"

        # (vii) True=0, Pred mixed (0/1), AUC = 0
        elif np.all(y_true_all == 0) and np.any(y_pred_all == 0) and np.any(y_pred_all == 1):
            auc_micro = 0.0
            fpr_micro = np.array([0.0, 1.0])
            tpr_micro = np.array([0.0, 0.0])
            msg = "True == 0 and PRED == 0/1 --> AUC = 0"

        # fallback (shouldn't happen)
        else:
            auc_micro = 0.5
            fpr_micro = np.array([0.0, 1.0])
            tpr_micro = np.array([0.0, 1.0])
            true_val = int(y_true_all[0])
            msg = f"Other case: True == {true_val}) --> AUC = 0.5"

        if verbose:
            print(msg)

    # --- CASE 2: normal ROC ---
    else:
        try:
            fpr_micro, tpr_micro, _ = metrics.roc_curve(y_true_all, y_pred_all)
            auc_micro = metrics.auc(fpr_micro, tpr_micro)
            if verbose:                
                print(f"Normal case --> AUC = {auc_micro:.4f}")
        except ValueError:
            fpr_micro = np.array([0.0, 1.0])
            tpr_micro = np.array([0.0, 0.0])
            auc_micro = 0.0
            if verbose:
                print("ROC computation failed --> AUC = 0")

    if verbose:
        print("\n" + "-" * 40)
        print(f"Micro AUC: {auc_micro:.4f}")
        print("=" * 40)

    return fpr_micro, tpr_micro, auc_micro


# ------------------------------------------------------------------------------#
#                          ROBUST WEIGHTED ROC AUC                              #
# ------------------------------------------------------------------------------#
def robust_weighted_roc_auc(true, pred, verbose=True):
    """
    Compute a robust, weighted-average ROC-AUC across multiple labels,
    handling special cases where ROC curves are undefined.

    --------------------------------------------------------------------------
    Description
    --------------------------------------------------------------------------
    For each label (column) in `true` and `pred`, this function:
      - Computes the ROC curve (FPR, TPR) and AUC in normal cases.
      - Handles edge cases where `y_true` contains only one class (all zeros
        or all ones), or where predictions are constant or binary-only.
      - Defines deterministic AUC values for these cases to ensure numerical
        stability and interpretability.
      - Aggregates individual label AUCs into a weighted-average score,
        using the number of positive samples as weights.

    --------------------------------------------------------------------------
    Objective
    --------------------------------------------------------------------------
    To provide a stable and interpretable computation of the weighted-average
    ROC-AUC for multi-label data, even when certain labels have no variance or
    when predicted scores are degenerate (constant or binary).
    The weighting ensures that labels with more positive samples have a
    proportionally greater influence on the final metric, making it suitable
    for imbalanced multi-label datasets.

    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    true : pandas.DataFrame
        Ground-truth binary indicators (0 or 1) for each label column.
        Each column represents one label, and all columns must have
        the same number of rows.

    pred : pandas.DataFrame
        Predicted scores or probabilities for each label column.
        Columns must match those in `true` and appear in the same order.
        May contain either binary {0, 1} predictions or continuous
        probabilities in [0, 1].

    verbose : bool, default=True
        If True, prints a per-label message describing the computed or
        assigned AUC, and displays the final weighted-average result.

    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------
    tuple: (fpr_dict, tpr_dict, auc_weighted, auc_df_detailed)

    fpr_dict : dict
        A dictionary mapping each label name to an array of False Positive
        Rates used to construct the ROC curve.  
        For special cases, standardized arrays (e.g., [0.0, 1.0]) are returned.

    tpr_dict : dict
        A dictionary mapping each label name to an array of True Positive
        Rates corresponding to the ROC curve.  
        For special cases, arrays such as [0.0, 1.0] or [0.0, 0.0] are used.

    auc_weighted : float
        The weighted mean of per-label AUCs, where weights are the proportion
        of positive samples per label (`support_i / total_support`).
        If total support is zero, equal weights are applied across labels.

    auc_df_detailed : pandas.DataFrame
        A detailed DataFrame containing one row per label with columns:
          - "Label": label name  
          - "AUC": computed or assigned AUC value  
          - "Support": number of positive samples in `true[col]`
          - "Weight": normalized support (used for weighted averaging)          
          - "FPR": array of False Positive Rates  
          - "TPR": array of True Positive Rates  

    --------------------------------------------------------------------------
    Special Case Handling Summary
    --------------------------------------------------------------------------
      - `y_true` all 0 and `y_pred` all 0  → AUC = 1.0  
      - `y_true` all 1 and `y_pred` all 1  → AUC = 1.0  
      - `y_true` all 0 and `y_pred` all 1  → AUC = 0.0  
      - `y_true` all 1 and `y_pred` all 0  → AUC = 0.0  
      - `y_true` constant (0 or 1) and `y_pred` probabilistic (not just 0/1)
        → AUC = 0.5  
      - `y_true` all 1, `y_pred` mixed {0,1} → AUC = 1.0  
      - `y_true` all 0, `y_pred` mixed {0,1} → AUC = 0.0  
      - ROC computation error (ValueError) → AUC = 0.0  
      - Any other fallback condition → AUC = 0.5  

    --------------------------------------------------------------------------
    Example
    --------------------------------------------------------------------------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.datasets import make_multilabel_classification
    >>> # Generate sample multilabel data
    >>> X, Y = make_multilabel_classification(
    ...     n_samples=100, n_features=10, n_classes=3, random_state=42)
    >>> true = pd.DataFrame(Y, columns=["Label1", "Label2", "Label3"])
    >>> # Simulate probabilistic predictions
    >>> rng = np.random.default_rng(42)
    >>> pred = pd.DataFrame({
    ...     "Label1": rng.random(100),
    ...     "Label2": rng.random(100),
    ...     "Label3": rng.random(100)
    ... })
    >>> fpr_dict, tpr_dict, auc_weighted, auc_df = robust_weighted_roc_auc(true, pred)
    >>> print(f"Weighted ROC-AUC: {auc_weighted:.4f}")
    >>> print(auc_df[["Label", "Support", "Weight", "AUC"]])
    Label   Support  Weight    AUC
    Label1     53.0   0.500   0.704
    Label2     31.0   0.292   0.668
    Label3     22.0   0.208   0.630
    """


    aucs = []
    supports = []
    fpr_dict = {}
    tpr_dict = {}

    if verbose:
        print("\n" + "="*40)
        print("WEIGHTED-AVERAGE ROC")
        print("="*40)

    for col in true.columns:
        y_true = true[col].values
        y_pred = pred[col].values
        support = np.sum(y_true)
        supports.append(support)

        # --- CASE 1: y_true has only one class (no variation) ---
        if len(np.unique(y_true)) < 2:
            
            # (i) both empty: True=0, Pred=0, AUC = 1
            if np.all(y_true == 0) and np.all(y_pred == 0):
                auc = 1.0
                fpr_dict[col] = np.array([0.0, 1.0])
                tpr_dict[col] = np.array([0.0, 1.0])
                msg = f"{col}: True == 0 and PRED == 0 --> AUC = 1"
            
            # (ii) both full: True=1, Pred=1, AUC = 1
            elif np.all(y_true == 1) and np.all(y_pred == 1):
                auc = 1.0
                fpr_dict[col] = np.array([0.0, 1.0])
                tpr_dict[col] = np.array([0.0, 1.0])
                msg = f"{col}: True == 1 and PRED == 1 --> AUC = 1"
            
            # (iii) True empty, Pred full, AUC = 0
            elif np.all(y_true == 0) and np.all(y_pred == 1):
                auc = 0.0
                fpr_dict[col] = np.array([0.0, 1.0])
                tpr_dict[col] = np.array([1.0, 1.0])
                msg = f"{col}: True == 0 and PRED == 1 --> AUC = 0"
            
            # (iv) True full, Pred empty, AUC = 0
            elif np.all(y_true == 1) and np.all(y_pred == 0):
                auc = 0.0
                fpr_dict[col] = np.array([0.0, 1.0])
                tpr_dict[col] = np.array([0.0, 0.0])
                msg = f"{col}: True == 1 and PRED == 0 --> AUC = 0"
            
            # (v) True all 0 or all 1, PRED has probabilities, AUC = 9.5
            elif len(np.unique(y_true)) == 1 and not np.all(np.isin(y_pred, [0, 1])):
                auc = 0.5
                fpr_dict[col] = np.array([0.0, 1.0])
                tpr_dict[col] = np.array([0.0, 1.0])
                true_val = int(y_true[0])
                msg = f"{col}: True == {true_val} and PRED probabilistic --> AUC = 0.5"

            # (vi) True=1, Pred mixed (0/1), AUC = 1
            elif np.all(y_true == 1) and np.any(y_pred == 0) and np.any(y_pred == 1):
                auc = 1.0
                fpr_dict[col] = np.array([0.0, 1.0])
                tpr_dict[col] = np.array([0.0, 1.0])
                msg = f"{col}: True == 1 and PRED == 0/1 --> AUC = 1"

            # (vii) True=0, Pred mixed (0/1), AUC = 0
            elif np.all(y_true == 0) and np.any(y_pred == 0) and np.any(y_pred == 1):
                auc = 0.0
                fpr_dict[col] = np.array([0.0, 1.0])
                tpr_dict[col] = np.array([0.0, 0.0])
                msg = f"{col}: True == 0 but PRED == 0/1 --> AUC = 0"

            # fallback (shouldn't happen)
            else:
                auc = 0.5
                fpr_dict[col] = np.array([0.0, 1.0])
                tpr_dict[col] = np.array([0.0, 1.0])
                msg = f"{col}: Other case --> AUC = 0.5"

        # --- CASE 2: normal case ---
        else:
            try:
                fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
                auc = metrics.auc(fpr, tpr)
                fpr_dict[col] = fpr
                tpr_dict[col] = tpr
                msg = f"{col}: Normal Case --> AUC={auc:.4f}"
            except ValueError:
                auc = 0.0
                fpr_dict[col] = np.array([0.0, 1.0])
                tpr_dict[col] = np.array([0.0, 0.0])
                msg = f"{col}: ROC computation failed --> AUC = 0"

        aucs.append(auc)
        if verbose:
            print(msg)

    # --- weights computation ---
    supports = np.array(supports, dtype=float)
    if np.sum(supports) == 0:
        weights = np.ones_like(supports) / len(supports)
    else:
        weights = supports / np.sum(supports)

    auc_weighted = float(np.nansum(np.array(aucs) * weights))

    # --- detailed DataFrame ---
    auc_df_detailed = pd.DataFrame({
        "Label": true.columns,
        "AUC": aucs,
        "Support": supports,
        "Weight": weights,        
        "FPR": [fpr_dict.get(col, np.array([])) for col in true.columns],
        "TPR": [tpr_dict.get(col, np.array([])) for col in true.columns],
    })

    if verbose:
        print("=" * 40)
        print(f"Weighted ROC AUC: {auc_weighted:.4f}")
        print("=" * 40)

    return fpr_dict, tpr_dict, auc_weighted, auc_df_detailed



# ------------------------------------------------------------------------------#
#                          ROBUST WEIGHTED ROC AUC                              #
# ------------------------------------------------------------------------------#
def robust_sample_roc_auc(true, pred, verbose=True):
    """
    Compute a robust sample-average ROC-AUC with special case handling and FPR/TPR tracking.

    --------------------------------------------------------------------------
    Description
    --------------------------------------------------------------------------
    This function computes the **samples-average ROC-AUC**, which evaluates
    the ROC-AUC per sample (row) and then averages the results.
    It is particularly relevant for multi-label classification, where each
    sample can belong to multiple classes.

    Each sample’s ROC curve (FPR, TPR) is computed individually.
    The function also handles degenerate cases robustly:
        - When all true labels or predictions for a sample are constant
        - When both are all zeros or all ones (perfect cases)
        - When true and predicted values are mismatched (inverted cases)

    --------------------------------------------------------------------------
    Objective
    --------------------------------------------------------------------------
    To compute a stable and interpretable per-sample ROC-AUC and summarize
    both the individual and average results, even when some samples have
    degenerate or constant values.

    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    true : pandas.DataFrame
        Binary ground-truth labels (0 or 1) for each class per sample.

    pred : pandas.DataFrame
        Predicted probabilities or binary predictions (0 or 1)
        with the same structure as `true`.

    verbose : bool, default=True
        If True, prints detailed information for each sample and overall stats.

    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------
    sample_auc_df : pandas.DataFrame
        DataFrame containing per-sample ROC-AUC results:
            - "Sample": name of the sample (e.g., Sample1, Sample2, ...)
            - "AUC": computed ROC-AUC value
            - "FPR": False Positive Rate array
            - "TPR": True Positive Rate array

    samples_auc_mean : float
        Mean AUC across all samples.

    --------------------------------------------------------------------------
    Example
    --------------------------------------------------------------------------
    >>> import pandas as pd
    >>> true = pd.DataFrame([[1, 0, 1], [0, 1, 0], [1, 1, 0]],
    ...                     columns=["Label1", "Label2", "Label3"])
    >>> pred = pd.DataFrame([[0.9, 0.2, 0.8], [0.1, 0.7, 0.3], [0.8, 0.9, 0.1]],
    ...                     columns=["Label1", "Label2", "Label3"])
    >>> sample_auc_df, samples_auc_mean = robust_sample_roc_auc(true, pred)
    >>> print(sample_auc_df)
        Sample    AUC             FPR             TPR
    0  Sample1  1.000  [0.0, 1.0]     [0.0, 1.0]
    1  Sample2  1.000  [0.0, 1.0]     [0.0, 1.0]
    2  Sample3  1.000  [0.0, 1.0]     [0.0, 1.0]
    >>> print(f"Samples average AUC = {samples_auc_mean:.4f}")
    """

    sample_results = []

    if verbose:
        print("\n" + "="*40)
        print("SAMPLES-AVERAGE ROC")
        print("="*40)

    for i in range(len(true)):
        y_true = true.iloc[i].values
        y_pred = pred.iloc[i].values

        # --- CASE 1: y_true has only one class (no variation) ---
        if len(np.unique(y_true)) < 2:
          
            # (i) both empty: True=0, Pred=0, AUC = 1
            if np.all(y_true == 0) and np.all(y_pred == 0):
                auc, fpr, tpr = 1.0, np.array([0.0, 1.0]), np.array([0.0, 1.0])
                msg = f"Sample {i+1}: True == 0 and Pred == 0 --> AUC = 1"
            
            # (ii) both full: True=1, Pred=1, AUC = 1
            elif np.all(y_true == 1) and np.all(y_pred == 1):
                auc, fpr, tpr = 1.0, np.array([0.0, 1.0]), np.array([0.0, 1.0])
                msg = f"Sample {i+1}: True == 1 and Pred == 1 --> AUC = 1"
            
            # (iii) True empty, Pred full, AUC = 0
            elif np.all(y_true == 0) and np.all(y_pred == 1):
                auc, fpr, tpr = 0.0, np.array([0.0, 1.0]), np.array([1.0, 1.0])
                msg = f"Sample {i+1}: True == 0 and Pred == 1 --> AUC = 0"
                
            # (iv) True full, Pred empty, AUC = 0
            elif np.all(y_true == 1) and np.all(y_pred == 0):
                auc, fpr, tpr = 0.0, np.array([0.0, 1.0]), np.array([0.0, 0.0])
                msg = f"Sample {i+1}: True == 1 and Pred == 0 --> AUC = 0"
            
            # (v) True all 0 or all 1, PRED has probabilities, AUC = 9.5
            elif len(np.unique(y_true)) == 1 and not np.all(np.isin(y_pred, [0, 1])):
                auc, fpr, tpr = 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])
                true_val = int(y_true[0])  # vai ser 0 ou 1
                msg = f"Sample {i+1}: True == {true_val} and PRED probabilistic --> AUC = 0.5"

            # (vi) True=1, Pred mixed (0/1), AUC = 1
            elif np.all(y_true == 1) and np.any(y_pred == 0) and np.any(y_pred == 1):                
                auc, fpr, tpr = 1.0, np.array([0.0, 1.0]), np.array([0.0, 0.0])                
                msg = f"Sample {i+1}: True == 1 but Pred == 0/1 --> AUC = 1"

            # (vii) True=0, Pred mixed (0/1), AUC = 0
            elif np.all(y_true == 0) and np.any(y_pred == 0) and np.any(y_pred == 1):
                auc, fpr, tpr = 0.0, np.array([0.0, 1.0]), np.array([0.0, 0.0])                                                
                msg = f"Sample {i+1}: True == 0 but Pred == 0/1 --> AUC = 0"

            # fallback (shouldn't happen)
            else:
                auc, fpr, tpr = 0.0, np.array([0.0, 1.0]), np.array([0.0, 0.0])                                                
                msg = f"Sample {i+1}: Other case --> AUC = 0.5"                

            if verbose:
                print(msg)
        else:
            try:
                fpr, tpr, _ = metrics.roc_curve(y_true.astype(float), y_pred.astype(float))
                auc = metrics.auc(fpr, tpr)
                if verbose:
                    print(f"Sample {i+1}: Normal case --> AUC = {auc:.4f}")
            except ValueError:
                auc, fpr, tpr = 0.0, np.array([0.0, 1.0]), np.array([0.0, 0.0])
                if verbose:
                    print(f"Sample {i+1}: ROC computation failed --> AUC = 0")

        sample_results.append((i, auc, fpr, tpr))

    # Build final DataFrame
    sample_auc_df = pd.DataFrame(sample_results, columns=["Index", "AUC", "FPR", "TPR"])
    sample_auc_df["Sample"] = ["Sample" + str(i + 1) for i in sample_auc_df["Index"]]
    sample_auc_df = sample_auc_df[["Sample", "AUC", "FPR", "TPR"]]

    samples_auc_mean = np.mean(sample_auc_df["AUC"])

    if verbose:
        print("\n" + "-"*40)
        print(f"Samples mean AUC: {samples_auc_mean:.4f}")
        print("="*40)

    return sample_auc_df, samples_auc_mean


# ---------------------------------------------------------------
# MACRO
# ---------------------------------------------------------------
def plot_macro_roc(fpr_dict, tpr_dict, macro_auc, macro_auc_interp, macro_df, save_path=None, show=False):
    """
    Plot macro-average ROC curve with optional saving and display.
    """
    plt.figure(figsize=(8, 6))
    
    # Curvas individuais
    for label in fpr_dict.keys():
        plt.plot(fpr_dict[label], tpr_dict[label], lw=1, alpha=0.7,
                 label=f"{label} (AUC={macro_df.loc[macro_df['Label']==label, 'AUC'].values[0]:.3f})")

    # Curva média
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(all_fpr, fpr_dict[label], tpr_dict[label]) for label in fpr_dict.keys()], axis=0)
    plt.plot(all_fpr, mean_tpr, color='red', lw=2.5, label=f"Mean (interp) AUC={macro_auc_interp:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Macro-average ROC Curve\nMean AUC={macro_auc:.3f}, Interp={macro_auc_interp:.3f}")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # === Salvar ===
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf", dpi=1200, bbox_inches="tight", transparent=True)
    
    # === Mostrar (opcional) ===
    if show:
        plt.show()
    else:
        plt.close()



# ---------------------------------------------------------------
# MICRO
# ---------------------------------------------------------------
def plot_micro_roc(fpr, tpr, auc_micro, save_path=None, show=False):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"Micro-average (AUC = {auc_micro:.3f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-average ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf", dpi=1200, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------
# WEIGHTED
# ---------------------------------------------------------------
def plot_weighted_roc(fpr_dict, tpr_dict, auc_weighted, auc_df, save_path=None, show=False):
    plt.figure(figsize=(8, 6))
    for label in fpr_dict.keys():
        plt.plot(fpr_dict[label], tpr_dict[label], lw=1.5, alpha=0.7,
                 label=f"{label} (AUC={auc_df.loc[auc_df['Label']==label, 'AUC'].values[0]:.3f}, W={auc_df.loc[auc_df['Label']==label, 'Weight'].values[0]:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Weighted-average ROC Curve (AUC = {auc_weighted:.3f})")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf", dpi=1200, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------
# SAMPLES
# ---------------------------------------------------------------
def plot_samples_auc(sample_auc_df, save_path=None, show=False):
    plt.figure(figsize=(8, 6))
    plt.hist(sample_auc_df["AUC"], bins=20, color="skyblue", edgecolor="black", alpha=0.8)
    plt.xlabel("Sample-wise ROC-AUC")
    plt.ylabel("Frequency")
    plt.title("Distribution of Sample-wise ROC-AUC")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf", dpi=1200, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    else:
        plt.close()