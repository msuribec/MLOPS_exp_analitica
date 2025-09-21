from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X, y, prefix="training_"):
    """
    Evaluate the model on the given dataset and return key metrics.

    Parameters
    ----------
    model : object
        The trained model with a predict method.
    X : pd.DataFrame or np.ndarray
        Features for evaluation.
    y : pd.Series or np.ndarray
        True labels.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, and F1-score.
    """
    y_pred = model.predict(X)

    metrics = {
        f"{prefix}accuracy": accuracy_score(y, y_pred),
        f"{prefix}precision": precision_score(y, y_pred),
        f"{prefix}recall": recall_score(y, y_pred),
        f"{prefix}f1_score": f1_score(y, y_pred)
    }

    return metrics