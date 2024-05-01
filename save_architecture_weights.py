import tensorflow as tf
from tf_keras.models import load_model

import json

model = load_model('CNN_model.keras')

with open('CNN_model.json', 'w') as json_file:
    json_file.write(model.to_json())

model.save_weights('CNN_model_weights.h5')