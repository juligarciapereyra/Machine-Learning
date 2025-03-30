# Proyecto Final ML: Speech Emotion Recognition (SER)

El modelo Dual se ejecuta unicamente corriendo el main y se pueden ver sus metricas y desempeño sobre el set de test. 

Como correr nuestros experimentos:

1. Definir los dataframes leyendo la direccion en donde se guardaron

        train_df = pd.read_csv(r'C:\Users\delfi\Desktop\SER_mod\data\unified_train_song_speech.csv', index_col=False)
        test_df = pd.read_csv(r'C:\Users\delfi\Desktop\SER_mod\data\unified_test_song_speech.csv', index_col=False)

2. Definir la grilla de parametros con los que se va a tunear el modelo. Se van a correr todas las combinaciones posibles
   
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [10, 20],
            'min_samples_split': [4, 8],
            'min_samples_leaf': [2, 4]
        }

4. Definir todos los experimentos que se quieren correr. Dentro de dataset pueden ir solo song, solo speech, ambos, con emociones de calm y speech unificadas, etc. Dentro de group_type pueden ir todas las maneras en las que se quiera correr el cross validation: 'both' es eligiendo un actor y un statement como validacion y entrenar con todos los datos que no son ni de ese actor ni de ese statement;  'actor' es eligiendo un actor para validar y entrenando con el resto; 'statement' es eligiendo una frase para validar y entrenando con la otra. Dentro de norm_type se puede poner None - sin normalizar - , 'global' - normalizar con media y desviacion por feature al set de train y luego aplicarle la misam normalizacion al set de test DENTRO DE CADA FOLD - , 'actor' es normalizar por feature de cada actor. ACLARACION: la normalizacion por actor es unicamente con fines experimentales para analizar el efecto de la individualidad de las emociones de los actores porque no se puede llevar a la practica en la vida real con el modelo en produccion. Se van a correr todas las combinaciones posibles de los elementos dentro del diccionario experiments y se buscaran los mejores hiperparametros para cada una utilizando todas las de param_grid.

experiments =  {
    'dataset': [(train_df, test_df, 'unified')],
    'group_type': ['both'],
    'norm_type': ['global'],  
}

4. Llamar a run_eperiments e indicarse el modelo, el diccionario de experimentos y el de parametros. Esperar a que tunee ;). 
run_experiments('rf', experiments, param_grid)

5. Poner el modelo en produccion :D

OBS: la mayoria de los hiperparametros que obtivmos se encuentran en configs
