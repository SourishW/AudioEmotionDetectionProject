import numpy as np
from tf_keras import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tf_keras.optimizers import Adam
import tensorflow as tf

import data_generators

output_classes = 7
lr = 1e-3

tf.random.set_seed(42)
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(data_generators.targ_height, data_generators.targ_width, 3)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
# cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dense(output_classes, activation='softmax'))

cnn.compile(
    optimizer=Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_data = data_generators.get_data()
validation_data = data_generators.get_data('validation')

cnn.fit(
    train_data,
    epochs=1,
    batch_size=data_generators.batch_size,
    validation_data=validation_data
)

loss, accuracy = cnn.evaluate(validation_data)

print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

cnn.save('CNN_model.keras')