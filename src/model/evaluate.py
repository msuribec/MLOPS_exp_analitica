from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X, y, prefix="training_"):
    """Evalua el modelo en los datos proporcionados y devuelve las métricas.
    Parameters
    ----------
    model : object
        Modelo entrenado que implementa los métodos predict y predict_proba.
    X : array-like
        Características de entrada para la evaluación.
    y : array-like
        Etiquetas verdaderas para la evaluación.
    prefix : str, optional
        Prefijo para las métricas devueltas (default: "training_").
    Returns
    -------
    dict
        Diccionario con las métricas de evaluación.
    """
    y_pred = model.predict(X)

    metrics = {
        f"{prefix}accuracy": accuracy_score(y, y_pred),
        f"{prefix}precision": precision_score(y, y_pred),
        f"{prefix}recall": recall_score(y, y_pred),
        f"{prefix}f1_score": f1_score(y, y_pred)
    }

    return metrics