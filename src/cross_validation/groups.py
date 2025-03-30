import numpy as np
import pandas as pd
from normalization.normalizer import DataNormalizer


def get_folds_per_actor(df, num_actors):
    
        folds_idx = [[] for _ in range(num_actors)]
        data_group = []
        for i in range(len(df)):
            sample_actor = int(df.iloc[i]['actor'] - 1)
            folds_idx[sample_actor].append(i)
            data_group.append(sample_actor)
        return folds_idx, data_group # devuelve num_actors listas, la primera con los indices de los datos correspondientes al primer actor, la segunda con los, segundos, etc. y una lista ordenada con el grupo correspondiente a cada dato
    

def get_folds_per_statement(df, num_statements):
    
        folds_idx = [[] for _ in range(num_statements)]
        data_group = []
        
        for i in range(len(df)):
            sample_statement = int(df.iloc[i]['statement'] - 1)
            folds_idx[sample_statement].append(i)
            data_group.append(sample_statement)
            
        return folds_idx, data_group


def get_folds_idx(df, group_type, num_actors, num_statements):    
    if group_type == 'actor':
        return get_folds_per_actor(df, num_actors)[0] # devuelve num_actors listas, la primera con los indices de los datos correspondientes al primer actor, la segunda con los segundos, etc.
    
    if group_type == 'statement':
        return get_folds_per_statement(df, num_statements)[0]
        
    # else default both
    groups = np.empty((num_actors, num_statements), dtype=object)
    
    for x in range(num_actors):
        for y in range(num_statements):
            groups[x][y] = []
            
    folds_idx_actor, data_group_actor = get_folds_per_actor(df, num_actors)
    folds_idx_statement, data_group_statement = get_folds_per_statement(df, num_statements)
    
    for i in range(len(df)):
        groups[data_group_actor[i]][data_group_statement[i]].append(i)
        
    return groups

   
def get_train_test_folds_idx(df, group_type, num_actors, num_statements):
    
    fold_idx = get_folds_idx(df, group_type, num_actors, num_statements)
    
    train_idx = []
    test_idx = []
    
    if group_type == 'actor':
        actor_valid = 0
        while actor_valid < num_actors:
            temp_train = []
            test_idx.append(fold_idx[actor_valid])
            for actor in range(num_actors):
                if actor != actor_valid:
                    temp_train.extend(fold_idx[actor])
            train_idx.append(temp_train)
            actor_valid += 1
                    
        return train_idx, test_idx
    
    elif group_type == 'statement':
        statement_valid = 0
        while statement_valid < num_statements:
            temp_train = []
            test_idx.append(fold_idx[statement_valid])
            for statement in range(num_statements):
                if statement != statement_valid:
                    temp_train.extend(fold_idx[statement])
            train_idx.append(temp_train)
            statement_valid += 1
            
        return train_idx, test_idx
    
    else: #both default
        for actor_valid in range(num_actors):
            for statement_valid in range(num_statements):
                temp_train = []
                test_idx.append(fold_idx[actor_valid][statement_valid])
                
                for actor in range(num_actors):
                    if actor != actor_valid:
                        temp_train.extend(fold_idx[actor][statement_valid])
                for statement in range(num_statements):
                    if statement != statement_valid:
                        temp_train.extend(fold_idx[actor_valid][statement])
                
                train_idx.append(temp_train)
                
        return train_idx, test_idx # al zippear tengo los indices de los grupos de train y test y voy iterando

   
def get_train_test_folds_data(df, group_type, norm_type=None, target='emotion', emotion=False):
    
    num_actors = len(np.unique(df['actor']))
    num_statements = len(np.unique(df['statement']))
    train_idx, test_idx = get_train_test_folds_idx(df, group_type, num_actors, num_statements)

    num_folds = len(train_idx)
    
    X_train_folds = [] 
    Y_train_folds = []
    
    X_test_folds = []
    Y_test_folds = []
    
    for fold in range(num_folds):
        
        df_train = df.iloc[train_idx[fold]]
        df_test = df.iloc[test_idx[fold]]
        
        if norm_type:
            
            normalizer = DataNormalizer()
            df_train = normalizer.normalize(df_train, norm_type=norm_type)
            df_test = normalizer.normalize(df_test, norm_type=norm_type, test_set=True)
            
        X_tr_fold = []
        Y_tr_fold = []
            
        X_tst_fold = []
        Y_tst_fold = []
            
        for i in range(len(df_train)):
            if target == 'intensity' and emotion:
                selected_columns = df_train.columns[:3].append(df_train.columns[df_train.columns.get_loc('F0semitoneFrom27.5Hz_sma3nz_amean'):])
                X_tr_fold.append(df_train.iloc[i][selected_columns].values)
                #X_tr_fold.append(df_train.iloc[i].loc['F0semitoneFrom27.5Hz_sma3nz_amean':].values)  
            else:
                X_tr_fold.append(df_train.iloc[i].loc['F0semitoneFrom27.5Hz_sma3nz_amean':].values)  
                
            Y_tr_fold.append(df_train.iloc[i, df_train.columns.get_loc(target)])

        for j in range(len(df_test)):
            if target == 'intensity' and emotion:
                selected_columns = df_test.columns[:3].append(df_test.columns[df_test.columns.get_loc('F0semitoneFrom27.5Hz_sma3nz_amean'):])
                X_tst_fold.append(df_test.iloc[j][selected_columns].values)
            else:
                X_tst_fold.append(df_test.iloc[j].loc['F0semitoneFrom27.5Hz_sma3nz_amean':].values)
            Y_tst_fold.append(df_test.iloc[j, df_test.columns.get_loc(target)])
                        

        X_train_folds.append(X_tr_fold)
        Y_train_folds.append(Y_tr_fold)
        X_test_folds.append(X_tst_fold)
        Y_test_folds.append(Y_tst_fold)
        
    return X_train_folds, Y_train_folds, X_test_folds, Y_test_folds 
