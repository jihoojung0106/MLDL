from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import xgboost as xgb
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import pandas as pd
import csv
data = pd.read_csv("train.csv")

X = data.drop('defects', axis=1)
y = data['defects']


param_grid = {
    'max_depth': [3,5],
    'learning_rate': [0.05,0.01,0.1],
    'n_estimators': [100,200,300,400],
    'subsample': [1,0.7,0.5],
    'alpha': [0.1, 0.2],
}

results = {}

best_hyperparameters = None
best_accuracy = 0
best_f1_score = 0

for max_depth in param_grid['max_depth']:
    for learning_rate in param_grid['learning_rate']:
        for n_estimators in param_grid['n_estimators']:
            for subsample in param_grid['subsample']:
                for alpha in param_grid['alpha']:
                    params = {
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators,
                        'subsample': subsample,
                        'alpha': alpha,
                        'objective': 'binary:logistic'
                    }

                    model = xgb.XGBClassifier(**params)
                    model.fit(X, y)

                    accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                    f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')

                    params_str = f"max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}, subsample={subsample}, alpha={alpha}"
                    
                    mean_accuracy = np.mean(accuracy_scores)
                    mean_f1_score = np.mean(f1_scores)
                    results[params_str] = {
                        'params': params,
                        'mean_accuracy': mean_accuracy,
                        'mean_f1_score': mean_f1_score
                    }
                    weighted_metric = 0.7 * mean_accuracy + 0.3 * mean_f1_score

                    if weighted_metric > 0.7 * best_accuracy + 0.3 * best_f1_score:
                        best_accuracy = mean_accuracy
                        best_f1_score = mean_f1_score
                        best_hyperparameters = params
                        print("***better***",params_str)

                    print(params_str)
                    print("Cross-Validation Accuracy Scores:", accuracy_scores)
                    print("Cross-Validation Weighted F1 Scores:", f1_scores)
                    print("Mean Accuracy:", mean_accuracy)
                    print("Mean Weighted F1 Score:", mean_f1_score)
                    print("Weighted Metric:", weighted_metric)
                    print("\n\n")


print("Best Hyperparameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)
print("Best Weighted F1 Score:", best_f1_score)


'''parameter results'''
data_list = []
for params_str, result in results.items():
    max_depth = result['params']['max_depth']
    learning_rate = result['params']['learning_rate']
    n_estimators = result['params']['n_estimators']
    subsample = result['params']['subsample']
    alpha = result['params']['alpha']
    accuracy = result['mean_accuracy']
    f1_score = result['mean_f1_score']

    data_list.append([max_depth, learning_rate, n_estimators, subsample, alpha, accuracy, f1_score])

column_names = ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'alpha', 'accuracy', 'F1 score']
csv_file_path = 'model_results.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(column_names)  
    writer.writerows(data_list)    
print(f'Results saved to {csv_file_path}')

'''test data'''
test_data = pd.read_csv("test.csv")
best_model = xgb.XGBClassifier(**best_hyperparameters)
best_model.fit(X, y)
y_pred = best_model.predict(test_data)
predictions_df = pd.DataFrame({'defects': y_pred})
predictions_df['defects'] = predictions_df['defects'].replace({0: False, 1: True})

output_filename = "2019-17577_pred.csv"
predictions_df.to_csv(output_filename, index=False,header=False)
