import numpy as np
from sklearn.metrics import accuracy_score

def equal_opportunity(y_true, y_pred, sensitive):
    """
    Computes True Positive Rate (TPR) per group.
    """
    results = {}
    for g in np.unique(sensitive):
        mask = (sensitive == g) & (y_true == 1)
        results[int(g)] = np.mean(y_pred[mask])
    return results

def demographic_parity(y_pred, sensitive):
    """
    Computes positive prediction rate per group.
    """
    results = {}
    for g in np.unique(sensitive):
        results[int(g)] = np.mean(y_pred[sensitive == g])
    return results

def evaluate_model(model, X, y, sensitive):
    """
    Returns accuracy, EO gap, and DP gap.
    """
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    eo = equal_opportunity(y, y_pred, sensitive)
    dp = demographic_parity(y_pred, sensitive)

    eo_gap = abs(eo[1] - eo[2])
    dp_gap = abs(dp[1] - dp[2])

    return {
        "accuracy": acc,
        "eo": eo,
        "eo_gap": eo_gap,
        "dp": dp,
        "dp_gap": dp_gap
    }
