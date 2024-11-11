#importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Loading the dataset
data = pd.read_csv('emails.csv')

# Data Preprocessing
# Dropping irrelevant columns
data = data.drop(['Email No.'], axis=1)  # Drop any non-feature columns

# Handling missing values if any
data.fillna(0, inplace=True)

# Splitting features and target
X = data.drop('Prediction', axis=1)
y = data['Prediction']

# Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test spliting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#We use 2 rounds of search, first with randomSearchCV for larger hyperparameter space and then with GridSearchCV for narrow, more focused and refined space!
# First Round: Coarse Parameter Search
rf_param_grid_coarse = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_classifier = RandomForestClassifier(random_state=42)

# Running Randomized Search for initial coarse tuning
print("Running Coarse Randomized Search on Random Forest...")
rf_random_search_coarse = RandomizedSearchCV(estimator=rf_classifier, param_distributions=rf_param_grid_coarse, 
                                             n_iter=10, cv=5, scoring='f1', verbose=1, random_state=42, n_jobs=-1)
rf_random_search_coarse.fit(X_train, y_train)

# Results from coarse tuning
best_params_coarse = rf_random_search_coarse.best_params_
print("Best parameters after coarse Randomized Search:", best_params_coarse)
print("Best F1 Score (coarse Random Search):", rf_random_search_coarse.best_score_)

# Refinement of Hyperparameter Ranges based on Initial Results
# Narrowing down around the best parameters found in the coarse search
rf_param_grid_refined = {
    'n_estimators': [best_params_coarse['n_estimators'] - 20, best_params_coarse['n_estimators'], best_params_coarse['n_estimators'] + 20],
    'max_depth': [best_params_coarse['max_depth'] - 10 if best_params_coarse['max_depth'] is not None else None,
                  best_params_coarse['max_depth'], 
                  best_params_coarse['max_depth'] + 10 if best_params_coarse['max_depth'] is not None else None],
    'min_samples_split': [max(2, best_params_coarse['min_samples_split'] - 1), best_params_coarse['min_samples_split'], best_params_coarse['min_samples_split'] + 1],
    'min_samples_leaf': [max(1, best_params_coarse['min_samples_leaf'] - 1), best_params_coarse['min_samples_leaf'], best_params_coarse['min_samples_leaf'] + 1],
    'bootstrap': [best_params_coarse['bootstrap']]
}

# Running Grid Search with refined parameter grid
print("\nRunning Refined Grid Search on Random Forest...")
rf_grid_search_refined = GridSearchCV(estimator=rf_classifier, param_grid=rf_param_grid_refined, cv=5, 
                                      scoring='f1', verbose=1, n_jobs=-1)
rf_grid_search_refined.fit(X_train, y_train)

# Results from refined tuning
best_params_refined = rf_grid_search_refined.best_params_
print("Best parameters after refined Grid Search:", best_params_refined)
print("Best F1 Score (refined Grid Search):", rf_grid_search_refined.best_score_)

# Final Evaluation on Test Set using Refined Model
best_rf_model_refined = rf_grid_search_refined.best_estimator_
print("\nEvaluation on Test Set - Refined Model")
rf_predictions_refined = best_rf_model_refined.predict(X_test)
print(classification_report(y_test, rf_predictions_refined))
