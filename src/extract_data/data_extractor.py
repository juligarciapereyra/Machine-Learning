import os
import pandas as pd
import opensmile
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

def get_file_paths(data_dir):
    file_paths = []  
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_paths.append(os.path.join(root, file))
    return file_paths


def get_metadata(data_dir):
    """
    Filename identifiers:
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). (There is no strong intensity for the 'neutral' emotion).
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    """
    file_paths = get_file_paths(data_dir)
    data = []
    for path in file_paths:
        file_name = os.path.basename(path)  
        parts = file_name.split('-')
        vocal_channel = int(parts[1])
        emotion = int(parts[2])
        intensity = int(parts[3])
        statement = int(parts[4])
        repetition = int(parts[5])
        actor = int(parts[6].split('.')[0])
        gender = 'male' if actor % 2 == 1 else 'female'

        data.append([path, vocal_channel, emotion, intensity, statement, repetition, actor, gender])
        
    return data


def extract_features(data):
    
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    features_list = []
    columns = None
    
    for entry in data:
        path = entry[0]
        features = smile.process_file(path)
        
      
        features_row = features.iloc[0].tolist()
        if columns is None:
            columns = features.columns.tolist()
        features_list.append(entry + features_row)
    
    metadata_columns = ['path', 'vocal_channel', 'emotion', 'intensity', 'statement', 'repetition', 'actor', 'gender']
    df = pd.DataFrame(features_list, columns=metadata_columns + columns)
    
    return df


def split_data_by_actor(df, train_actors=22):
    """
    Split the dataframe into training and testing sets based on actor.

    Args:
        df (pd.DataFrame): DataFrame containing metadata and features.
        train_actors (int): Number of actors to include in the training set.
    
    Returns:
        train_df (pd.DataFrame): Training set.
        test_df (pd.DataFrame): Testing set.
    """
    train_df = df[df['actor'] <= train_actors]
    test_df = df[df['actor'] > train_actors]
    
    return train_df, test_df


def get_data_sets(data_dir, save_dir, train_file, test_file):
    data = get_metadata(data_dir)
    df = extract_features(data)
    train_df, test_df = split_data_by_actor(df)

    train_csv_path = os.path.join(save_dir, train_file)
    test_csv_path = os.path.join(save_dir, test_file)

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    return df, train_df, test_df


def unify_calm_neutral(df):
    new_df = df.copy()
    new_df.loc[new_df['emotion'] == 2, 'emotion'] = 1 # calm and neutral are now emotion 1
    new_df.loc[new_df['emotion'] > 2, 'emotion'] -= 1 # decrease numbers of the rest of the emotions (to get 1-7 labels); neutral stays as 1 with calm unified.
    return new_df


def unify_emotions(save_dir, train, test, song=False):
        train = unify_calm_neutral(train)
        test = unify_calm_neutral(test)

        train_path = os.path.join(save_dir, f'unified_train_speech.csv')
        test_path = os.path.join(save_dir, f'unified_test_speech.csv')

        if song:
            train_path = os.path.join(save_dir, f'unified_train_song_speech.csv')
            test_path = os.path.join(save_dir, f'unified_test_song_speech.csv')

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        return train, test


def convert_to_df(train, test): 
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)
    return train_df, test_df


def polish_features(df): 
    Y = df['emotion']
    X = df.drop(columns=['emotion', 'path', 'vocal_channel', 'intensity', 'statement', 'repetition', 'actor', 'gender'])

    return X, Y


def get_sets_for_intensity(df, emotions = True, oversample=False, desired_ratio=2.6):
    
    df = df[~df['emotion'].isin([1, 2, 6, 7])]
    
    if oversample:
        positive_samples = df[df['intensity'] == 2]  
        negative_samples = df[df['intensity'] == 1]

        negative_count = len(negative_samples)
        desired_positive_count = int(negative_count * desired_ratio)

        selected_positive_samples = positive_samples.sample(n=desired_positive_count, replace=True, random_state=42)
        oversampled_df = shuffle(pd.concat([selected_positive_samples, negative_samples]), random_state=42)
        df = oversampled_df.copy()
    
    Y = (df['intensity']).values.astype(np.float32)
    X = df.loc[:,'F0semitoneFrom27.5Hz_sma3nz_amean':]
    
    columns = df.columns
    X = X.values.astype(np.int8)
    
    if not emotions:
        return X, Y, df
    
    onehot_encoder = OneHotEncoder(sparse_output=False)
    emotion_onehot = onehot_encoder.fit_transform(df[['emotion']])
    X_emotion = np.hstack((emotion_onehot, X))

    df_emotion = pd.DataFrame(np.hstack((emotion_onehot, df.values)))
    df_emotion.columns = np.concatenate(([0,1,2], columns))
    
    return X_emotion, Y, df_emotion