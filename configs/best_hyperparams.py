
# best hyperparams for emotion classifiers
params_gb = {'n_estimators': 300, 'learning_rate': 0.105, 'max_depth': 5, 'min_samples_leaf': 3}
best_params_RF = {'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 12, 'min_samples_leaf': 3}



# best hyperparams for intensity classifiers
params_lr ={'C': 2, 'solver': 'lbfgs', 'max_iter': 500, 'class_weight': {1: 1, 2: 2.6}}
params_gb_intensity = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 4, 'min_samples_leaf': 2}
params_mlp_intensity = {'activation': 'relu', 'hidden_layer_sizes': (64,), 'learning_rate_init': 0.0001, 'max_iter': 5000, 'solver': 'adam'}