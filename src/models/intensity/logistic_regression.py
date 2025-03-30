from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

class LogisticRegressionModel:
    def __init__(self, **kwargs):
        self.model = SklearnLogisticRegression(**kwargs, random_state=42)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    