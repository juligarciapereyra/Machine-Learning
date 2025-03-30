from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier

class MLPClassifierModelintensity:
    def __init__(self, **kwargs):
        self.model = SklearnMLPClassifier(**kwargs, random_state=42)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)