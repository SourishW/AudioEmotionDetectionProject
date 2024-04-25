import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow import keras
from keras import layers
from keras import models
from scikeras.wrappers import KerasClassifier

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

# Reshape X to fit the CNN input shape (number of samples, width, height, channels)
X = X.reshape(X.shape[0], 12, 1023, 1)  # Assuming 25 chunks and 511 frequencies per chunk

# # Normalize the data
# scaler = MinMaxScaler()
# X_normalized = scaler.fit_transform(X.reshape(-1, 25 * 511 * 1))
# X_normalized = X_normalized.reshape(-1, 25, 511, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to create the CNN model
def create_model(optimizer='adam', filters=64, dense_units=64):
    # model = models.Sequential([
    #     layers.layers.Conv2D(filters, (3, 20), activation='relu', input_shape=(12, 1023, 1)),
    #     layers.layers.MaxPooling2D((2, 2)),
    #     layers.layers.Conv2D(filters, (5, 40), activation='relu'),
    #     layers.layers.MaxPooling2D((2, 2)),
    #     layers.Flatten(),
    #     layers.Dense(dense_units, activation='relu'),
    #     layers.Dense(1, activation='sigmoid')
    # ])

#     model = models.Sequential([
#     layers.layers.Conv2D(filters, (3, 20), activation='relu', input_shape=(12, 1023, 1), padding='same'),
#     layers.layers.MaxPooling2D((2, 2)),
#     layers.layers.Conv2D(filters, (3, 40), activation='relu', padding='same'),
#     layers.layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(dense_units, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
#     model.compile(optimizer=optimizer,
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(12, 1023, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Create a KerasClassifier with the create_model function
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define the hyperparameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
}


# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print the best parameters and best accuracy
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Use the best parameters to create and train the final model
best_params = grid_result.best_params_
final_model = create_model(**best_params, filters=32, dense_units=64)  # Specify filters and dense_units here
final_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the final model
test_loss, test_acc = final_model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

