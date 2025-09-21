from sklearn.ensemble import RandomForestClassifier

class rfClassifier:
    def __init__(self, args):
        """
        Initialize the RandomForestClassifier with any scikit-learn parameters.
        Example:
            model = rfClassifier(n_estimators=200, max_depth=10)
        """
        self.model = RandomForestClassifier(**args)

    def fit(self, X, y):
        """Fit the model on training data."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict labels for input features."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for input features."""
        return self.model.predict_proba(X)

    def score(self, X, y):
        """Return accuracy of the model on given data."""
        return self.model.score(X, y)
