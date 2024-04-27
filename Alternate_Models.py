import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow import keras
from keras import layers
from keras import models
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

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

    # # Normalize the data
    # scaler = MinMaxScaler()
    # X_normalized = scaler.fit_transform(X.reshape(-1, 25 * 511 * 1))
    # X_normalized = X_normalized.reshape(-1, 25, 511, 1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Random Forest:", accuracy)


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


    # Create and train Decision Tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = dt_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Decision Tree Classifier Accuracy:", accuracy)





# Define the main method
def main():
    
    RandomForest()
    
    DecisionTrees()
    

# Check if this script is being run directly
if __name__ == "__main__":
    main()