from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from normalization.normalizer import DataNormalizer
from extract_data.data_extractor import polish_features



class KNN:
    def __init__(self, **kwargs):
        self.model = SklearnKNeighborsClassifier(**kwargs)
        self.normalizer = DataNormalizer()
        
    def fit(self, train, norm_type):
        df_train = train
        if norm_type is not None:
            df_train = self.normalizer.normalize(train, norm_type)
        X_train, Y_train = polish_features(df_train)
        self.model.fit(X_train, Y_train)
    
    def predict(self, test, norm_type):
        df_test = test
        if norm_type is not None:
            df_test = self.normalizer.normalize(test, norm_type, test_set=True)
        X_test, Y_test = polish_features(df_test)
        predictions = self.model.predict(X_test)

        return Y_test, predictions
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)