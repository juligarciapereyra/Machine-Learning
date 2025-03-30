import sys
sys.path.append(r'C:\Users\delfi\Desktop\SER_def\src')
from extract_data.data_extractor import get_data_sets, unify_calm_neutral, split_data_by_actor, polish_features
from models.final_model.dual_model import DualModel
import pandas as pd
from src.metrics.metrics_functions import get_scores, display_metrics_as_table
import numpy as np


def get_speech_song_df(speech_dir, save_speech_dir, song_dir, save_song_dir, speech_train_file, speech_test_file, song_train_file, song_test_file):
    
    df_speech, train_df_speech, test_df_speech = get_data_sets(speech_dir, save_speech_dir, speech_train_file, speech_test_file)
    df_song, train_df_song, test_df_song = get_data_sets(song_dir, save_song_dir, song_train_file, song_test_file)
    df = (pd.concat([df_speech, df_song], ignore_index=True))
    
    return df

def main(speech_dir, save_speech_dir, song_dir, save_song_dir, speech_train_file, speech_test_file, song_train_file, song_test_file):
    
    df = get_speech_song_df(speech_dir, save_speech_dir, song_dir, save_song_dir, speech_train_file, speech_test_file, song_train_file, song_test_file)
    df_unif = unify_calm_neutral(df)
    train_df, test_df = split_data_by_actor(df_unif)
    params_gb = {'n_estimators': 300, 'learning_rate': 0.105, 'max_depth': 5, 'min_samples_leaf': 3}
    params_lr ={'C': 2, 'solver': 'lbfgs', 'max_iter': 500, 'class_weight': {1: 1, 2: 2.6}}
    
    X_test, Y_test = polish_features(test_df)
    
    model = DualModel(params_gb, params_lr)

    model.fit(train_df)
    Y_true_emotion = test_df['emotion'].values
    Y_true_intensity = test_df['intensity'].values

    new_df, emotion_evaluation, intensity_evaluation = model.predict_evaluate_results(X_test, Y_true_emotion, Y_true_intensity)

    metrics_emotions = get_scores(emotion_evaluation['true_emotion'], emotion_evaluation['predicted_emotion'], emotion_labels = ['Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'])
    display_metrics_as_table(metrics_emotions, emotion_labels = ['Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'])
    metrics_intensity = get_scores(intensity_evaluation['true_intensity'], intensity_evaluation['predicted_intensity'], emotion_labels = ['0', '1'])
    display_metrics_as_table(metrics_intensity, emotion_labels = ['0', '1'])
    
if __name__ == '__main__':
    
    speech_dir = r'C:\Users\delfi\Desktop\ML_SER\data\Audio_Speech_Actors_01-24' # change for your route to speech audios
    save_speech_dir = r'C:\Users\delfi\Desktop\SER_mod\data'
    speech_train_file = 'ravdess_train_speech_set.csv'
    speech_test_file = 'ravdess_test_speech_sets.csv'
    song_dir = r'C:\Users\delfi\Desktop\ML_SER\data\Audio_Song_Actors_01-24' # change for yor route to song audios
    save_song_dir = r'C:\Users\delfi\Desktop\SER_mod\data'
    song_train_file = 'ravdess_train_song_set.csv'
    song_test_file = 'ravdess_test_song_sets.csv'

    main(speech_dir, save_speech_dir, song_dir, save_song_dir, speech_train_file, speech_test_file, song_train_file, song_test_file)
    
    
    