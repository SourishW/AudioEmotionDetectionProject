import numpy as np
from tf_keras import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.optimizers import Adam
from tf_keras.regularizers import l1_l2
import tensorflow as tf
import tf_keras

import data_generators

output_classes = 7
lr = 4e-4

l1 = 0.01
l2 = 0.00

dropout = 0.4

tf.random.set_seed(42)
np.random.seed(24)
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(data_generators.targ_height, data_generators.targ_width, 3), kernel_regularizer=l1_l2(l1, l2)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l1_l2(l1, l2)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l1_l2(l1, l2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu', kernel_regularizer=l1_l2(l1, l2)))
cnn.add(Dropout(dropout))
cnn.add(Dense(output_classes, activation='softmax', kernel_regularizer=l1_l2(l1, l2)))

cnn.compile(
    optimizer=Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_data = data_generators.get_data()
validation_data = data_generators.get_data('validation')

cnn.fit(
    train_data,
    epochs=15,
    batch_size=data_generators.batch_size,
    validation_data=validation_data
)

loss, accuracy = cnn.evaluate(validation_data)

print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

cnn.save('CNN_model.keras')