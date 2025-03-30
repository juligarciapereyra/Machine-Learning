import numpy as np
import pandas as pd


class DataNormalizer:
    def __init__(self):
        self.global_means = None
        self.global_stds = None
        self.actor_means = {}
        self.actor_stds = {}
    
    def fit_global(self, data, features):
        self.global_means = data[features].mean()
        self.global_stds = data[features].std()

    def normalize_global(self, data, features):
        if self.global_means is None or self.global_stds is None:
            print("Global normalization needs to be fitted first")
            return -1
        normalized_data = data.copy()
        for feature in features:
            normalized_data[feature] = (data[feature] - self.global_means[feature]) / self.global_stds[feature]
        return normalized_data
    
    def fit_actor(self, data, features): # fits mean and std for each actor, stores it in dictionary
        actors = data['actor'].unique()
        for actor in actors:
            actor_data = data[data['actor'] == actor]
            self.actor_means[actor] = actor_data[features].mean()
            self.actor_stds[actor] = actor_data[features].std()
    
    def normalize_by_actor(self, data, features):
        normalized_data = data.copy()
        actors = data['actor'].unique()
        for actor in actors:
            actor_data = normalized_data[normalized_data['actor'] == actor].copy()
            for feature in features:
                normalized_data.loc[normalized_data['actor'] == actor, feature] = (
                    (actor_data[feature] - self.actor_means[actor][feature]) / self.actor_stds[actor][feature]
                )
        return normalized_data

    def normalize(self, df, norm_type='global', test_set=False, features=None):
        
        if features is None:  
            features = df.loc[:,'F0semitoneFrom27.5Hz_sma3nz_amean':].columns  
        
        if norm_type == 'actor':
            self.fit_actor(df, features)
            return self.normalize_by_actor(df, features)
        
        if not test_set:
            self.fit_global(df, features) # fit means and std of train set
            
        return self.normalize_global(df, features)