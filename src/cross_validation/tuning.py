import numpy as np
from itertools import product # cartesian product to get all the combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from cross_validation.groups import get_train_test_folds_data
from models.emotions.mlp import MLPClassifierModel
from models.emotions.knn import KNN
from models.intensity.logistic_regression import LogisticRegressionModel
from models.intensity.gradient_boosting_intensity import GradientBoostingModel
from models.intensity.mlp_intensity import MLPClassifierModelintensity

def get_param_combs(param_grid):
    param_combinations = []
    keys = param_grid.keys()
    values = param_grid.values()

    for combination in product(*values):
        param_dict = dict(zip(keys, combination))
        param_combinations.append(param_dict)
    
    return param_combinations


def tune_hyperparameters(model_name, df, group_type, param_grid, norm_type, epochs=20, print_epochs=False, target='emotion', emotion=False):
 
    param_combinations = get_param_combs(param_grid)
    best_acc = 0
    best_params = None

    X_train_folds, Y_train_folds, X_test_folds, Y_test_folds = get_train_test_folds_data(df, group_type=group_type, norm_type=norm_type, target=target, emotion=emotion)

    y_true = None
    y_pred = None
    
    for params in tqdm(param_combinations, desc=f"Tuning hyperparameters for {model_name}"):

        y_true_params = []
        y_preds_params = []

        for i in range(len(X_train_folds)):
        
            X_train_fold, X_test_fold = X_train_folds[i], X_test_folds[i]
            Y_train_fold, Y_test_fold = Y_train_folds[i], Y_test_folds[i]
            if len(Y_test_fold) >= 2: # to catch case of Actor 18 with no data in song statements
                if model_name == 'rf':
                    model = RandomForestClassifier(**params, random_state=42, class_weight='balanced_subsample')
                    model.fit(X_train_fold, Y_train_fold)
                elif model_name == 'gb' and target != 'intensity':
                    model = GradientBoostingClassifier(**params, random_state=42)
                    model.fit(X_train_fold, Y_train_fold)
                elif model_name == 'mlp' and target != 'intensity': # emotion mlp
                    model = MLPClassifierModel(params)
                    model.fit(X_train_fold, Y_train_fold, X_val=X_test_fold, Y_val=Y_test_fold, epochs=epochs, print_epochs=print_epochs)
                elif model_name == 'knn':
                    model = KNN(**params)
                    model.fit(X_train_fold, Y_train_fold)
                elif model_name == 'lr':
                    model = LogisticRegressionModel(**params)
                    model.fit(X_train_fold, Y_train_fold)
                elif (model_name == 'gb' and target == 'intensity') or model_name=='gb intensity': # intensity gb
                    model = GradientBoostingModel(**params)
                    model.fit(X_train_fold, Y_train_fold)
                elif (model_name == 'mlp' and target == 'intensity') or model_name=='mlp intensity': # intensity mlp
                    model = MLPClassifierModelintensity(**params)
                    model.fit(X_train_fold, Y_train_fold)
                else:
                    print("Not a valid model. Please try again")
                    return -1
 
                if (model_name == 'mlp' and target == 'intensity') or model_name=='mlp intensity':
                    y_pred = model.predict(np.array(X_test_fold).astype(np.float32))
                else:
                    y_pred = model.predict(X_test_fold)
                      
                y_preds_params.extend(y_pred)
                y_true_params.extend(Y_test_fold)
          
          

        acc = accuracy_score(y_true_params, y_preds_params)
        
        if acc > best_acc:
            best_acc = acc
            best_params = params
            
            y_true = y_true_params
            y_preds = y_preds_params
    
    return best_acc, best_params, y_true, y_preds
    
    
    
    
    
    
    
    
