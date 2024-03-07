from sklearn import metrics
import numpy as np


def get_cm_values(y_true, y_pred, num_classes):
    MCM = metrics.multilabel_confusion_matrix(y_true, y_pred, labels=range(num_classes))
    tn = MCM[:, 0, 0]
    fn = MCM[:, 1, 0]
    tp = MCM[:, 1, 1]
    fp = MCM[:, 0, 1]
    return tn, fn, tp, fp

def get_multi_class_dp(y_pred_list, group_list, num_classes):
    dp_group1 = np.zeros(num_classes)
    dp_group2 = np.zeros(num_classes)

    for class_idx in range(num_classes):
        dp_group1[class_idx] = (y_pred_list[group_list==0]==class_idx).sum() / (y_pred_list[group_list==0]).size
        dp_group2[class_idx] = (y_pred_list[group_list==1]==class_idx).sum() / (y_pred_list[group_list==1]).size

    dp = np.asarray([group1 - group2 for (group1, group2) in zip(dp_group1, dp_group2)])
    return dp

def _prf_divide(numerator, denominator, zero_division=0):
    """ Performs division and handles divide-by-zero. (Copyed from sklearn)
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1 # avoid infs/nans
    result = numerator / denominator
    if not np.any(mask):
        return result
    result[mask] = 0.0 if zero_division in [0] else 1.0
    return result

def compute_fairness_metrics(label_list, y_pred_list, group_list, zero_division=0):
    num_classes = max(label_list.max(), y_pred_list.max())+1
    tn, fn, tp, fp = get_cm_values(label_list, y_pred_list, num_classes)
    # just want to get the weighted sum
    true_sum = tp + fn
    weights = true_sum
    if sum(group_list==0) > 0:
        tn, fn, tp, fp = get_cm_values(label_list[group_list==0], y_pred_list[group_list==0], num_classes)
        sklearn_TPR_group1 = _prf_divide(tp, tp + fn, zero_division)
        sklearn_TNR_group1 = _prf_divide(tn, tn + fp, zero_division)
        sklearn_FPR_group1 = _prf_divide(fp, tn + fp, zero_division)
        sklearn_TR_group1 = _prf_divide(tp + tn, tp + fp + tn + fn, zero_division)
    if sum(group_list==1) > 0:
        tn, fn, tp, fp = get_cm_values(label_list[group_list==1], y_pred_list[group_list==1], num_classes)
        sklearn_TPR_group2 = _prf_divide(tp, tp + fn, zero_division)
        sklearn_TNR_group2 = _prf_divide(tn, tn + fp, zero_division)
        sklearn_FPR_group2 = _prf_divide(fp, tn + fp, zero_division)
        sklearn_TR_group2 = _prf_divide(tp + tn, tp + fp + tn + fn, zero_division)
    # equalized opportunity
    equal_opportunity_gap_y0 = np.average(sklearn_TNR_group1-sklearn_TNR_group2)
    equal_opportunity_gap_y1 = np.average(sklearn_TPR_group1-sklearn_TPR_group2)
    equal_opportunity_gap_y0_abs = np.average(np.abs(sklearn_TNR_group1-sklearn_TNR_group2))
    equal_opportunity_gap_y1_abs = np.average(np.abs(sklearn_TPR_group1-sklearn_TPR_group2))

    # demographic_parity
    demographic_parity_distance = np.sum(get_multi_class_dp(y_pred_list, group_list, num_classes))
    demographic_parity_distance_abs = np.sum(np.abs(get_multi_class_dp(y_pred_list, group_list, num_classes)))

    # equalized odds
    equal_odds = ((sklearn_TPR_group1 - sklearn_TPR_group2) + (sklearn_FPR_group1 - sklearn_FPR_group2)).mean() / 2
    equal_odds_abs = np.abs((sklearn_TPR_group1 - sklearn_TPR_group2) + (sklearn_FPR_group1 - sklearn_FPR_group2)).mean() / 2

    # Statistical parity difference
    statistical_parity_difference = np.abs(sklearn_TR_group1 - sklearn_TR_group2).mean()

    fairness_metric = {'fairness/DP': demographic_parity_distance, 'fairness/EOpp0': equal_opportunity_gap_y0, 
                       'fairness/EOpp1': equal_opportunity_gap_y1, 'fairness/EOdds': equal_odds,
                       'fairness/DP_abs': demographic_parity_distance_abs, 'fairness/EOpp0_abs': equal_opportunity_gap_y0_abs, 
                       'fairness/EOpp1_abs': equal_opportunity_gap_y1_abs, 'fairness/EOdds_abs': equal_odds_abs,
                       'fairness/SPD': statistical_parity_difference}

    return fairness_metric
