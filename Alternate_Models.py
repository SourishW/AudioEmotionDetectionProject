import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow import keras
from keras import layers
from keras import models
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

#RandomForestClassifier

def RandomForest():
    # Load your dataset
    df = pd.read_csv("emotion_data.csv")

    # Drop the index column
    df = df.drop(df.columns[0], axis=1)

    # Encode emotion labels
    label_encoder = LabelEncoder()
    df['emotion'] = label_encoder.fit_transform(df['emotion'])

    # Split the DataFrame into features (X) and labels (y)
    X = df.drop(['filename', 'emotion'], axis=1).values
    y = df['emotion'].values

   # Assuming you have your data loaded into X (features) and y (target)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Random Forest Classifier
    rf_clf = RandomForestClassifier()

    # Define parameter grid for Randomized Search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    # Define cross-validation strategy
    cv = 5

    # Define scoring function
    scoring = make_scorer(accuracy_score)

    # Perform Randomized Search Cross Validation
    random_search = RandomizedSearchCV(estimator = rf_clf, 
                                    param_distributions = param_grid, 
                                    n_iter = 30, 
                                    cv = cv, 
                                    scoring = scoring, 
                                    verbose = 2, 
                                    random_state = 42, 
                                    n_jobs = -1)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Get best parameters
    best_params = random_search.best_params_

    # Train the model with best parameters
    best_rf_clf = RandomForestClassifier(**best_params)
    best_rf_clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = best_rf_clf.predict(X_train)
    y_pred_test = best_rf_clf.predict(X_test)

    # Evaluate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Best Parameters:", best_params)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)


#Decision Trees

def DecisionTrees():
    
    # Load your dataset
    df = pd.read_csv("emotion_data.csv")

    # Drop the index column
    df = df.drop(df.columns[0], axis=1)

    # Encode emotion labels
    label_encoder = LabelEncoder()
    df['emotion'] = label_encoder.fit_transform(df['emotion'])

    # Split the DataFrame into features (X) and labels (y)
    X = df.drop(['filename', 'emotion'], axis=1).values
    y = df['emotion'].values

    # # Normalize the data
    # scaler = MinMaxScaler()
    # X_normalized = scaler.fit_transform(X.reshape(-1, 25 * 511 * 1))
    # X_normalized = X_normalized.reshape(-1, 25, 511, 1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Define Decision Tree Classifier
    dt_clf = DecisionTreeClassifier()

    # Define scoring function
    scoring = make_scorer(accuracy_score)

    # Define parameter grid for Grid Search
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    # Perform Grid Search Cross Validation
    grid_search = GridSearchCV(estimator=dt_clf,
                            param_grid=param_grid,
                            cv=5,
                            scoring=scoring,
                            verbose=2,
                            n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get best parameters
    best_params = grid_search.best_params_

    # Train the model with best parameters
    best_dt_clf = DecisionTreeClassifier(**best_params)
    best_dt_clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = best_dt_clf.predict(X_train)
    y_pred_test = best_dt_clf.predict(X_test)

    # Evaluate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Best Parameters:", best_params)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)


#KNN

def KNN():
    
    # Load your dataset
    df = pd.read_csv("emotion_data.csv")

    # Drop the index column
    df = df.drop(df.columns[0], axis=1)

    # Encode emotion labels
    label_encoder = LabelEncoder()
    df['emotion'] = label_encoder.fit_transform(df['emotion'])

    # Split the DataFrame into features (X) and labels (y)
    X = df.drop(['filename', 'emotion'], axis=1).values
    y = df['emotion'].values

    # # Normalize the data
    # scaler = MinMaxScaler()
    # X_normalized = scaler.fit_transform(X.reshape(-1, 25 * 511 * 1))
    # X_normalized = X_normalized.reshape(-1, 25, 511, 1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Define Decision Tree Classifier
    knn_clf = KNeighborsClassifier()
        
    # Define scoring function
    scoring = make_scorer(accuracy_score)

        # Define parameter grid for Grid Search
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40, 50],
        'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }

    # Perform Grid Search Cross Validation
    grid_search = GridSearchCV(estimator=knn_clf,
                            param_grid=param_grid,
                            cv=5,
                            scoring='accuracy',
                            verbose=2,
                            n_jobs=-1)

    # Fit the model on the smaller dataset
    grid_search.fit(X_train, y_train)

    # Get best parameters
    best_params = grid_search.best_params_

    # Train the model with best parameters on the full dataset
    best_knn_clf = KNeighborsClassifier(**best_params)
    best_knn_clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = best_knn_clf.predict(X_train)
    y_pred_test = best_knn_clf.predict(X_test)

    # Evaluate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Best Parameters:", best_params)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

#XGBoost
def XGBoost():
    
    # Load your dataset
    df = pd.read_csv("emotion_data.csv")

    # Drop the index column
    df = df.drop(df.columns[0], axis=1)

    # Encode emotion labels
    label_encoder = LabelEncoder()
    df['emotion'] = label_encoder.fit_transform(df['emotion'])

    # Split the DataFrame into features (X) and labels (y)
    X = df.drop(['filename', 'emotion'], axis=1).values
    y = df['emotion'].values

    # # Normalize the data
    # scaler = MinMaxScaler()
    # X_normalized = scaler.fit_transform(X.reshape(-1, 25 * 511 * 1))
    # X_normalized = X_normalized.reshape(-1, 25, 511, 1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define XGBoost Classifier
    xgb_clf = xgb.XGBClassifier()

    # Define parameter grid for Randomized Search
    param_grid = {
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0, 0.1],
    }

    # Perform Randomized Search Cross Validation
    random_search = RandomizedSearchCV(estimator=xgb_clf,
                                    param_distributions=param_grid,
                                    n_iter=10,
                                    cv=5,
                                    scoring='accuracy',
                                    verbose=2,
                                    random_state=42,
                                    n_jobs=-1)

    # Fit the model on the smaller dataset
    random_search.fit(X_train, y_train)

    # Get best parameters
    best_params = random_search.best_params_

    # Train the model with best parameters on the full dataset
    best_xgb_clf = xgb.XGBClassifier(**best_params)
    best_xgb_clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = best_xgb_clf.predict(X_train)
    y_pred_test = best_xgb_clf.predict(X_test)

    # Evaluate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Best Parameters:", best_params)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)


# Define the main method
def main():
    
    #RandomForest()
    
    #DecisionTrees()
    
    #KNN()
    
    XGBoost()               #add more models and implement new techniques to improve performance if desired

# Check if this script is being run directly
if __name__ == "__main__":
    main()