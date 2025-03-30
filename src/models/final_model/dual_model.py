import numpy as np
from models.intensity.logistic_regression import LogisticRegressionModel
from models.emotions.gradient_boosting import GradientBoostingClassifierModel
from extract_data.data_extractor import get_sets_for_intensity
from normalization.normalizer import DataNormalizer
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


class DualModel():
    def __init__(self, params_emotion, params_intensity):
        self.normalizer = DataNormalizer()
        self.emotion_classifier = GradientBoostingClassifierModel(params_emotion)
        self.intensity_classifier = LogisticRegressionModel(**params_intensity)
        self.features = None

    def fit_gb(self, df):
        self.emotion_classifier.fit(df)

    def fit_lr(self, X, Y):
        self.intensity_classifier.fit(X, Y)

    def fit(self, df):
        self.features = df.loc[:,'F0semitoneFrom27.5Hz_sma3nz_amean':].columns
        df_train_normed = self.normalizer.normalize(df, norm_type='global', test_set=False) # fits normalizer with train set
        X_train_emotions, Y_train_emotions, df_train_normed_emotions = get_sets_for_intensity(df_train_normed, emotions = True) # to predict with emotions given
        self.fit_gb(df_train_normed) # fit gradient boosting
        self.fit_lr(X_train_emotions, Y_train_emotions) # fit logistic regression
        
    def predict(self, X):
       
        X = (X - self.normalizer.global_means) / self.normalizer.global_stds
        predicted_emotions = self.emotion_classifier.model.predict(X)
        
        new_df = pd.DataFrame(np.hstack((predicted_emotions.reshape(-1, 1), X)))
        new_df.columns = np.concatenate((['emotion'], self.features))
        red_df = new_df[new_df['emotion'].isin([3,4,5])].copy()

        X = red_df.loc[:,'F0semitoneFrom27.5Hz_sma3nz_amean':]
        columns = X.columns
        onehot_encoder = OneHotEncoder(sparse_output=False)
        emotion_onehot = onehot_encoder.fit_transform(red_df[['emotion']])
        X_emotion = np.hstack((emotion_onehot, X))

        df_emotion = pd.DataFrame(np.hstack((emotion_onehot, X.values)))
        df_emotion.columns = np.concatenate(([0,1,2], columns))
        
        predicted_intensity = self.intensity_classifier.predict(X_emotion)
        
        return new_df, predicted_emotions, predicted_intensity

                        
    def predict_evaluate_results(self, X, Y_true_emotion, Y_true_intensity):
        _, predicted_emotions, _ = self.predict(X)
    
        emotion_evaluation = pd.DataFrame({'true_emotion': Y_true_emotion, 'predicted_emotion': predicted_emotions})
        
        new_df = pd.DataFrame(np.hstack((predicted_emotions.reshape(-1, 1), X)))

        new_df.columns = np.concatenate((['emotion'], self.features))

        emotion_filtered_df = new_df[new_df['emotion'].isin([3, 4, 5])]
        emotion_filtered_idx = emotion_filtered_df.index
        
        filtered_true_emotion = Y_true_emotion[emotion_filtered_idx]
        filtered_true_intensity = Y_true_intensity[emotion_filtered_idx]
        
        intensity_evaluation = pd.DataFrame({
            'true_emotion': filtered_true_emotion,
            'true_intensity': filtered_true_intensity
        })

        X_red = emotion_filtered_df.loc[:, 'F0semitoneFrom27.5Hz_sma3nz_amean':]
        
        onehot_encoder = OneHotEncoder(sparse_output=False)
        emotion_onehot = onehot_encoder.fit_transform(emotion_filtered_df[['emotion']])
        X_emotion = np.hstack((emotion_onehot, X_red))
        
        predicted_intensity = self.intensity_classifier.predict(X_emotion)
        intensity_evaluation['predicted_intensity'] = predicted_intensity
        
        return new_df, emotion_evaluation, intensity_evaluation
