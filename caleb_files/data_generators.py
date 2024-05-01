import tensorflow as tf
from tf_keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

data_directory = './img_unsized/'
targ_width = 256
targ_height = 256
batch_size = 64

def get_data(data_class='training'):
    return datagen.flow_from_directory(
        data_directory,
        target_size=(targ_height, targ_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset=data_class
    )
