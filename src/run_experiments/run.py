import itertools
from src.cross_validation.tuning import tune_hyperparameters
from metrics.metrics_functions import get_scores, display_metrics_as_table
from models.emotions.random_forest import RFClassifierModel
from models.emotions.gradient_boosting import GradientBoostingClassifierModel
from models.emotions.knn import KNN
from models.emotions.mlp import MLPClassifierModel
from models.intensity.logistic_regression import LogisticRegressionModel
from models.intensity.gradient_boosting_intensity import GradientBoostingModel
from models.intensity.mlp_intensity import MLPClassifierModelintensity
from normalization.normalizer import DataNormalizer
from extract_data.data_extractor import get_sets_for_intensity
import numpy as np

def run_experiments(model_name, experiments, param_grid, target='emotion', emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'], emotion_labels_unified = ['Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']):
    
    for dataset, group_type, norm_type in itertools.product(experiments['dataset'], experiments['group_type'], experiments['norm_type']):
        train_df, test_df, type = dataset
        
        if type == 'emotion':
            emotion = True
        else:
            emotion=False

        if type == 'all':
            dataset_name = "Original"
            emotion_labels_current = emotion_labels
            if model_name == 'mlp':
                param_grid['ouput_dim'] = [8] # correct last layer

        elif type == 'unified':
            dataset_name = "Unified Calm/Neutral"
            emotion_labels_current = emotion_labels_unified
            if model_name == 'mlp':
                param_grid['ouput_dim'] = [7] # correct last layer
        
        elif target=='intensity':
            dataset_name = type
            emotion_labels_current = [0, 1] # binary classification
            
        else:
            raise ValueError("Unknown dataset configuration")
        
        if (model_name != 'rf' and model_name != 'gb' and model_name != 'mlp' and model_name != 'lr' and model_name != 'gb intensity' and model_name != 'mlp intensity' and model_name != 'knn'):
            print('Unknown model. Please try again')
            return -1
        
        best_acc, best_params, y_true, y_pred = tune_hyperparameters(model_name, train_df, group_type, param_grid, norm_type, target=target, emotion=emotion)

        print(f"---------Results of {model_name} with cross validation for hyperparameter tuning---------")
        print(f"--> Combination: dataset={dataset_name}, group_type={group_type}, norm_type={norm_type}")
        print("~ Best parameters:", best_params)
        print(f"Accuracy: {best_acc}")

        scores = get_scores(y_true, y_pred, emotion_labels_current)
        display_metrics_as_table(scores, emotion_labels_current)

        if (norm_type == 'global') or (norm_type == None): # get metrics on test set
            print(f"Results of best {model_name} for test")
            
            if model_name == 'rf':
                model = RFClassifierModel(best_params)
                model.fit(train_df, norm_type)
                Y_test, predictions = model.predict(test_df, norm_type)
                
            elif model_name == 'gb' and target != 'intensity': # gradient boosting for emotions
                model = GradientBoostingClassifierModel(best_params)
                model.fit(train_df, norm_type)
                Y_test, predictions = model.predict(test_df, norm_type)
                
            elif model_name == 'knn':
                model = KNN(**best_params)
                model.fit(train_df, norm_type)
                Y_test, predictions = model.predict(test_df, norm_type)
                
            elif target=='intensity':
                
                if model_name == 'lr':
                    model = LogisticRegressionModel(**best_params) 
                elif model_name == 'gb' or model_name == 'gb intensity':
                    model = GradientBoostingModel(**best_params) 
                elif model_name == 'mlp' or model_name == 'mlp intensity':
                    model = MLPClassifierModelintensity(**best_params)
                else:
                    print('Unknown model')
                    return -1
                    
                normalizer = DataNormalizer()
                normed_data = normalizer.normalize(train_df, norm_type='global')
                normed_test = normalizer.normalize(test_df, 'global', test_set=True)
        
                if type == 'emotion': 
                    X_train, Y_train, _ = get_sets_for_intensity(normed_data, emotions= True)
                    X_test, Y_test, _ = get_sets_for_intensity(normed_test, emotions= True)
                else:    
                    X_train, Y_train, _ = get_sets_for_intensity(normed_data, emotions= False)
                    X_test, Y_test, _ = get_sets_for_intensity(normed_test, emotions= False) 
                    
                if model_name == 'mlp' or model_name == 'mlp intensity':
                    X_train, Y_train = np.array(X_train).astype(np.float32), np.array(Y_train).astype(np.float32)
                    X_test, Y_test = np.array(X_test).astype(np.float32), np.array(Y_test).astype(np.float32)

                model.fit(X_train, Y_train) 
                predictions = model.predict(X_test)

                            
            else:
                model = MLPClassifierModel(best_params)
                normalizer = DataNormalizer()
                normed_data = normalizer.normalize(train_df, norm_type=norm_type, test_set=False)
                X_train, Y_train = (normed_data.loc[:,'F0semitoneFrom27.5Hz_sma3nz_amean':]).values.astype(np.float32), normed_data['emotion'].values
                normed_test = normalizer.normalize(test_df, norm_type, test_set=True)
                X_test, Y_test = (normed_test.loc[:,'F0semitoneFrom27.5Hz_sma3nz_amean':]).values.astype(np.float32), normed_test['emotion'].values

                model.fit(X_train, Y_train)
                predictions = model.predict(X_test)
            
    
            metrics = get_scores(Y_test, predictions, emotion_labels_current)
            display_metrics_as_table(metrics, emotion_labels_current)
            print('\n\n')
        
        
def run_main_model(path):
    pass

