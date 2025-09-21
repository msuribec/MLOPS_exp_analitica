from sklearn.ensemble import RandomForestClassifier

class rfClassifier:
    def __init__(self, args):
        """
        Inicializa el clasificador RandomForest con los parámetros proporcionados.
        Parameters
        ----------
        args : dict
            Diccionario de parámetros para RandomForestClassifier.
        Returns
        -------
        None
        """
        self.model = RandomForestClassifier(**args)

    def fit(self, X, y):
        """Ajusta el modelo a los datos de entrenamiento."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predice las clases para las características de entrada."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predice las probabilidades de las clases para las características de entrada."""
        return self.model.predict_proba(X)

    def score(self, X, y):
        """Calcula la precisión del modelo en los datos proporcionados."""
        return self.model.score(X, y)
