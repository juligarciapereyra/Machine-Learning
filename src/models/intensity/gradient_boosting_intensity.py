from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier

class GradientBoostingModel:
    def __init__(self, **kwargs):
        self.model = SklearnGradientBoostingClassifier(**kwargs, random_state=42)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    